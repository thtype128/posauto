import requests
import re
import random
from collections import defaultdict
from typing import List, Dict, Any

OLLAMA_MODEL = "exaone3.5"
OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_answer_with_context(context_chunks: List[Dict], user_question: str) -> str:
    """
    다중 문서의 관련 문장들을 기반으로 Ollama에게 질문하여 답변 생성
    개선: 문서별 컨텍스트 구분 및 출처 명시 강화
    """
    if not context_chunks:
        return "❌ 관련 문서를 찾을 수 없습니다."
    
    # 문서별로 컨텍스트 그룹화
    document_contexts = defaultdict(list)
    for chunk in context_chunks:
        doc_name = chunk.get("document_name", chunk.get("source_collection", "알 수 없음"))
        document_contexts[doc_name].append(chunk)
    
    # 컨텍스트 텍스트 구성 (문서별로 구분)
    context_text = ""
    total_chunks_used = 0
    
    for doc_name, chunks in document_contexts.items():
        if total_chunks_used >= 8:  # 최대 8개 문단으로 제한
            break
            
        context_text += f"\n=== 📄 {doc_name} ===\n"
        
        # 각 문서에서 상위 점수 순으로 정렬
        chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        for i, chunk in enumerate(chunks[:3]):  # 문서당 최대 3개 문단
            if total_chunks_used >= 8:
                break
                
            score = chunk.get("score", 0)
            page = chunk.get("page", "?")
            text = chunk.get("text", "")
            
            context_text += f"[문단 {total_chunks_used + 1}, p.{page}, 관련도: {score:.2f}]\n{text}\n\n"
            total_chunks_used += 1
    
    # 프롬프트 구성
    prompt = (
        f"You are a helpful assistant for technical document analysis. "
        f"You have access to information from multiple documents.\n\n"
        f"Document excerpts:\n{context_text}\n"
        f"User Question: {user_question}\n\n"
        f"Instructions:\n"
        f"1. Answer based on the provided document excerpts\n"
        f"2. When referencing information, mention the specific document and page (e.g., 'According to Document A, page 5...')\n"
        f"3. If information comes from multiple documents, clearly distinguish the sources\n"
        f"4. If the answer requires information not in the excerpts, state this clearly\n"
        f"5. Synthesize information across documents when relevant\n"
        f"6. Use your technical knowledge to provide context and explanations\n"
        f"7. Be concise but comprehensive\n\n"
        f"Answer:"
    )
    
    try:
        response = requests.post(
            OLLAMA_URL, 
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 600  # 다중 문서 답변을 위해 약간 증가
                }
            },
            timeout=35
        )
        response.raise_for_status()
        
        answer = response.json()["response"].strip()
        
        # 답변 후처리: 출처 정보 보강
        if answer and len(document_contexts) > 1:
            answer += f"\n\n📚 **참조된 문서**: {', '.join(document_contexts.keys())}"
        
        return answer if answer else "❌ 답변을 생성할 수 없습니다."
        
    except requests.exceptions.Timeout:
        return "⏰ 답변 생성 시간이 초과되었습니다. 다시 시도해주세요."
    except requests.exceptions.ConnectionError:
        return "🔌 Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요."
    except Exception as e:
        print(f"[Ollama 오류] {e}")
        return f"❌ 답변 생성 중 오류가 발생했습니다: {str(e)}"

def summarize_user_question(user_question: str) -> str:
    """
    사용자 질문의 의도를 요약하여 다중 문서 검색에 최적화된 키워드로 변환
    개선: 다중 문서 환경에 맞는 키워드 확장
    """
    system_prompt = (
        "You are a semantic analyzer for multi-document technical search. "
        "Extract the core search intent and expand it with related technical terms.\n\n"
        "Examples:\n"
        "- '이 구성품은 어떤 일을 하나요?' → '구성품 기능 역할 용도'\n"
        "- '밸브는 왜 필요한가요?' → '밸브 필요성 목적 기능'\n"
        "- '두 설비의 차이점은?' → '설비 차이점 비교 특징'\n"
        "- '안전 기준은 무엇인가요?' → '안전 기준 규격 요구사항'\n"
        "- 'What are the operating conditions?' → 'operating conditions parameters specifications'\n"
        "- 'How do these systems compare?' → 'system comparison differences features'\n\n"
        f"User Question: {user_question}\n"
        f"Expanded Search Keywords:"
    )
    
    try:
        response = requests.post(
            OLLAMA_URL, 
            json={
                "model": OLLAMA_MODEL,
                "prompt": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 60
                }
            },
            timeout=10
        )
        response.raise_for_status()
        
        summary = response.json()["response"].strip()
        print(f"🧠 질문 요약: {user_question} → {summary}")
        return summary if summary else user_question
        
    except Exception as e:
        print(f"[질문 요약 오류] {e}")
        return user_question

def rank_cross_document_results(question: str, results: List[Dict], top_k: int = 10) -> List[Dict]:
    """
    다중 문서에서 온 검색 결과를 Ollama로 재순위화
    문서 간 정보의 보완성과 관련성을 종합 고려
    """
    if not results or len(results) <= top_k:
        return results
    
    # 문서별 결과 분석
    doc_analysis = defaultdict(list)
    for result in results:
        doc_name = result.get("document_name", result.get("source_collection", "Unknown"))
        doc_analysis[doc_name].append(result)
    
    # 컨텍스트 구성 (상위 20개까지만)
    context = ""
    for i, result in enumerate(results[:20]):
        doc_name = result.get("document_name", "Unknown")
        page = result.get("page", "?")
        text = result.get("text", "")
        score = result.get("score", 0)
        
        context += f"[{i}] {doc_name} p.{page} (score: {score:.3f})\n{text[:120]}...\n\n"
    
    prompt = (
        f"You are evaluating cross-document search results for relevance and complementarity.\n\n"
        f"Question: {question}\n\n"
        f"Available documents: {', '.join(doc_analysis.keys())}\n\n"
        f"Search results:\n{context}\n\n"
        f"Select the {top_k} most relevant result indices that:\n"
        f"1. Best answer the question\n"
        f"2. Provide complementary information from different documents\n"
        f"3. Maintain diversity across document sources when possible\n\n"
        f"Return only the indices as comma-separated numbers (e.g., 1,5,7,12,15):"
    )
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 50
                }
            },
            timeout=20
        )
        response.raise_for_status()
        
        result_text = response.json()["response"]
        
        # 인덱스 추출 및 검증
        indices = []
        for num_str in re.findall(r'\b\d+\b', result_text):
            idx = int(num_str)
            if 0 <= idx < len(results):
                indices.append(idx)
        
        # 선택된 결과들 반환
        if indices:
            selected_results = [results[i] for i in indices[:top_k]]
            
            # 문서 다양성 확인 및 로깅
            selected_docs = set(r.get("document_name", "Unknown") for r in selected_results)
            print(f"🎯 LLM 재순위화: {len(selected_results)}개 결과, {len(selected_docs)}개 문서")
            
            return selected_results
        else:
            return results[:top_k]
            
    except Exception as e:
        print(f"[교차 문서 순위 매기기 오류] {e}")
        return results[:top_k]

def generate_document_comparison(documents_info: Dict[str, Any], user_question: str) -> str:
    """
    여러 문서의 정보를 비교 분석하여 답변 생성
    """
    if len(documents_info) < 2:
        return "문서 비교를 위해서는 최소 2개의 문서가 필요합니다."
    
    comparison_context = ""
    for doc_name, info in documents_info.items():
        comparison_context += f"\n=== {doc_name} ===\n"
        comparison_context += f"내용: {info.get('content', '')}\n"
        comparison_context += f"페이지: {info.get('page', 'N/A')}\n"
    
    prompt = (
        f"You are analyzing and comparing information from multiple technical documents.\n\n"
        f"Documents to compare:\n{comparison_context}\n\n"
        f"User Question: {user_question}\n\n"
        f"Please provide a comparative analysis that:\n"
        f"1. Identifies similarities and differences\n"
        f"2. Highlights unique aspects of each document\n"
        f"3. Synthesizes the information to answer the question\n"
        f"4. Notes any conflicting information\n\n"
        f"Comparative Analysis:"
    )
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            },
            timeout=30
        )
        response.raise_for_status()
        
        return response.json()["response"].strip()
        
    except Exception as e:
        print(f"[문서 비교 오류] {e}")
        return "문서 비교 분석 중 오류가 발생했습니다."

def check_ollama_connection() -> bool:
    """
    Ollama 서버 연결 상태 확인 및 모델 상태 체크
    """
    try:
        # 1. 서버 기본 연결 확인
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return False
        
        # 2. 사용 중인 모델 확인
        models = response.json().get("models", [])
        model_names = [model.get("name", "") for model in models]
        
        # 우리가 사용하는 모델이 있는지 확인
        if any(OLLAMA_MODEL in name for name in model_names):
            return True
        
        # 모델이 없는 경우 다른 사용 가능한 모델 확인
        return len(models) > 0
        
    except:
        return False

def get_available_models() -> List[str]:
    """
    사용 가능한 Ollama 모델 목록 반환
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model.get("name", "") for model in models]
    except:
        pass
    return []
