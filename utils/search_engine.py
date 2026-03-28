from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict, Any

class MultiDocumentSearchEngine:
    """
    다중 문서를 지원하는 ChromaDB와 SentenceTransformer 통합 검색 엔진
    """
    
    def __init__(self, chroma_client, collection_names: List[str]):
        self.chroma_client = chroma_client
        self.collection_names = collection_names
        self.collections = {}
        self.all_documents_data = []  # 모든 문서의 데이터를 통합 저장
        self._sentence_model = None
        
        # 컬렉션들 로드
        self._load_collections()
        
    def _get_sentence_model(self):
        """SentenceTransformer 모델 지연 로딩"""
        if self._sentence_model is None:
            print("🔄 SentenceTransformer 모델 로딩 중...")
            self._sentence_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        return self._sentence_model
    
    def _load_collections(self):
        """선택된 컬렉션들을 로드하고 통합 데이터 구성"""
        self.collections = {}
        self.all_documents_data = []
        
        for collection_name in self.collection_names:
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
                self.collections[collection_name] = collection
                
                # 해당 컬렉션의 모든 데이터 가져오기
                all_data = collection.get()
                
                # 통합 데이터에 추가 (소스 컬렉션 정보 포함)
                for i, (doc, metadata) in enumerate(zip(all_data["documents"], all_data["metadatas"])):
                    enhanced_metadata = metadata.copy()
                    enhanced_metadata["source_collection"] = collection_name
                    enhanced_metadata["unified_index"] = len(self.all_documents_data)
                    
                    self.all_documents_data.append({
                        "text": doc,
                        "metadata": enhanced_metadata
                    })
                    
                print(f"✅ 컬렉션 '{collection_name}' 로드됨: {len(all_data['documents'])}개 문단")
                
            except Exception as e:
                print(f"❌ 컬렉션 '{collection_name}' 로드 실패: {e}")
    
    def search_chromadb(self, query: str, top_k: int = 10) -> List[Dict]:
        """ChromaDB 다중 컬렉션 검색"""
        all_results = []
        
        for collection_name, collection in self.collections.items():
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=min(top_k, collection.count())  # 컬렉션 크기 제한 고려
                )
                
                # 결과 처리
                for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                    all_results.append({
                        "text": doc,
                        "page": meta["page"],
                        "original_index": meta.get("original_index", 0),
                        "score": 1 - distance,  # 거리를 유사도로 변환
                        "source_collection": collection_name,
                        "context": self._get_context_window_by_collection(collection_name, meta.get("original_index", 0))
                    })
                    
            except Exception as e:
                print(f"❌ 컬렉션 '{collection_name}' 검색 실패: {e}")
        
        # 점수 순으로 정렬하여 상위 k개 반환
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]
    
    def search_sentence_transformer(self, query: str, top_k: int = 10) -> List[Dict]:
        """SentenceTransformer 기반 다중 문서 검색"""
        if not self.all_documents_data:
            return []
        
        model = self._get_sentence_model()
        texts = [doc["text"] for doc in self.all_documents_data]
        
        # 임베딩 계산
        sentence_embeddings = model.encode(texts, convert_to_tensor=True)
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # 코사인 유사도 계산
        scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings).squeeze()
        top_indices = scores.argsort(descending=True)[:top_k]
        
        search_results = []
        for idx in top_indices:
            idx_int = int(idx)
            doc_data = self.all_documents_data[idx_int]
            metadata = doc_data["metadata"]
            
            search_results.append({
                "text": doc_data["text"],
                "page": metadata["page"],
                "original_index": metadata.get("original_index", idx_int),
                "score": float(scores[idx]),
                "source_collection": metadata["source_collection"],
                "context": self._get_context_window_by_collection(
                    metadata["source_collection"], 
                    metadata.get("original_index", 0)
                )
            })
        
        return search_results
    
    def search_hybrid(self, query: str, top_k: int = 10) -> List[Dict]:
        """하이브리드 검색: ChromaDB + SentenceTransformer 결과 통합"""
        # 각각 더 많이 검색한 후 통합
        k_expanded = min(top_k * 2, 20)  # 확장된 검색
        
        chroma_results = self.search_chromadb(query, k_expanded)
        sentence_results = self.search_sentence_transformer(query, k_expanded)
        
        # 결과 통합 (중복 제거)
        all_results = {}
        
        # ChromaDB 결과 추가 (가중치 0.4)
        for result in chroma_results:
            key = f"{result['source_collection']}_{result['page']}_{result['original_index']}"
            result_copy = result.copy()
            result_copy["score"] *= 0.4
            result_copy["search_method"] = "chromadb"
            all_results[key] = result_copy
        
        # SentenceTransformer 결과 추가 (가중치 0.6)
        for result in sentence_results:
            key = f"{result['source_collection']}_{result['page']}_{result['original_index']}"
            if key in all_results:
                # 중복된 경우 점수 평균 및 메서드 표시
                all_results[key]["score"] = (all_results[key]["score"] + result["score"] * 0.6) / 2
                all_results[key]["search_method"] = "hybrid"
            else:
                result_copy = result.copy()
                result_copy["score"] *= 0.6
                result_copy["search_method"] = "sentence_transformer"
                all_results[key] = result_copy
        
        # 문서 다양성을 고려한 최종 선택
        final_results = self._diversify_results(list(all_results.values()), top_k)
        return final_results
    
    def search(self, query: str, method: str = "chromadb", top_k: int = 10) -> List[Dict]:
        """통합 검색 인터페이스"""
        if not self.collections:
            return []
        
        if method == "chromadb":
            return self.search_chromadb(query, top_k)
        elif method == "sentencetransformer":
            return self.search_sentence_transformer(query, top_k)
        elif method == "하이브리드":
            return self.search_hybrid(query, top_k)
        else:
            # 기본값은 ChromaDB
            return self.search_chromadb(query, top_k)
    
    def _diversify_results(self, results: List[Dict], top_k: int) -> List[Dict]:
        """
        문서 간 다양성을 고려하여 결과 선택
        각 문서에서 균등하게 결과를 가져오도록 조정
        """
        # 점수 순으로 정렬
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # 컬렉션별로 분류
        collection_results = {}
        for result in results:
            collection = result["source_collection"]
            if collection not in collection_results:
                collection_results[collection] = []
            collection_results[collection].append(result)
        
        # 각 컬렉션에서 균등하게 선택
        selected_results = []
        collection_names = list(collection_results.keys())
        
        # 라운드 로빈 방식으로 선택
        max_per_collection = max(1, top_k // len(collection_names)) if collection_names else 0
        
        for i in range(top_k):
            if not collection_names:
                break
                
            collection_idx = i % len(collection_names)
            current_collection = collection_names[collection_idx]
            
            if collection_results[current_collection]:
                selected_results.append(collection_results[current_collection].pop(0))
            
            # 해당 컬렉션의 결과가 모두 소진되었거나 개수 제한에 도달한 경우
            collection_count = len([r for r in selected_results if r["source_collection"] == current_collection])
            if not collection_results[current_collection] or collection_count >= max_per_collection:
                collection_names.remove(current_collection)
        
        return selected_results
    
    def _get_context_window_by_collection(self, collection_name: str, original_index: int, window: int = 1) -> str:
        """특정 컬렉션에서 주변 문단을 포함한 컨텍스트 생성"""
        # 해당 컬렉션의 문서만 필터링
        collection_docs = [doc for doc in self.all_documents_data 
                          if doc["metadata"]["source_collection"] == collection_name]
        
        if not collection_docs:
            return ""
        
        # original_index로 해당 문단 찾기
        target_doc = None
        target_idx = -1
        for i, doc in enumerate(collection_docs):
            if doc["metadata"].get("original_index", 0) == original_index:
                target_doc = doc
                target_idx = i
                break
        
        if target_doc is None:
            return ""
        
        # 컨텍스트 윈도우 생성
        start = max(0, target_idx - window)
        end = min(len(collection_docs), target_idx + window + 1)
        
        context_texts = []
        for i in range(start, end):
            text = collection_docs[i]["text"]
            if i == target_idx:
                context_texts.append(f"**{text}**")  # 중심 문단 강조
            else:
                context_texts.append(text)
        
        return " ... ".join(context_texts)
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """다중 문서 통계 정보 반환"""
        stats = {
            "total_collections": len(self.collections),
            "total_documents": len(self.all_documents_data),
            "collections_info": {}
        }
        
        # 컬렉션별 통계
        for collection_name in self.collection_names:
            collection_docs = [doc for doc in self.all_documents_data 
                             if doc["metadata"]["source_collection"] == collection_name]
            
            if collection_docs:
                pages = [doc["metadata"]["page"] for doc in collection_docs]
                stats["collections_info"][collection_name] = {
                    "document_count": len(collection_docs),
                    "page_range": f"{min(pages)}-{max(pages)}",
                    "total_pages": max(pages),
                }
        
        return stats
    
    def search_within_collection(self, collection_name: str, query: str, method: str = "chromadb", top_k: int = 10) -> List[Dict]:
        """특정 컬렉션 내에서만 검색"""
        if collection_name not in self.collections:
            return []
        
        # 임시로 단일 컬렉션 검색 엔진 생성
        temp_engine = MultiDocumentSearchEngine(self.chroma_client, [collection_name])
        return temp_engine.search(query, method, top_k)
