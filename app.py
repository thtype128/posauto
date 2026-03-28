import streamlit as st
import fitz  # PDF 파일 열기용
import re    # 정규표현식 기반 문단 분할용
import hashlib  # 컬렉션 이름 해싱용
from unidecode import unidecode  # 유니코드 정규화용
from chromadb import Client
from chromadb.config import Settings
from utils.pdf_utils import extract_paragraphs_with_meta, build_is_meaningless_checker, get_document_statistics
from utils.search_engine import MultiDocumentSearchEngine
from utils.ollama_utils import generate_answer_with_context, summarize_user_question, check_ollama_connection, rank_cross_document_results
import json
from datetime import datetime
import os

# ChromaDB 설정
CHROMA_DB_DIR = "./chromadb_storage"
METADATA_FILE = "./document_metadata.json"

def init_chromadb():
    """ChromaDB 클라이언트 초기화"""
    if not os.path.exists(CHROMA_DB_DIR):
        os.makedirs(CHROMA_DB_DIR)
    return Client(Settings(persist_directory=CHROMA_DB_DIR))

def sanitize_collection_name(file_path: str) -> str:
    """ChromaDB 컬렉션 이름을 유효한 형식으로 변환하고 고유성을 보장"""
    file_name = file_path.split("\\")[-1].replace(".pdf", "")
    sanitized_name = unidecode(file_name)
    sanitized_name = re.sub(r'[^a-zA-Z0-9._-]', '_', sanitized_name)
    if len(sanitized_name) < 3:
        sanitized_name = "default_collection"
    if not sanitized_name[0].isalnum():
        sanitized_name = f"a{sanitized_name}"
    if not sanitized_name[-1].isalnum():
        sanitized_name = f"{sanitized_name}z"
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    sanitized_name = f"{sanitized_name}_{file_hash}"
    return sanitized_name

def load_document_metadata():
    """문서 메타데이터 로드"""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_document_metadata(metadata):
    """문서 메타데이터 저장"""
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def add_document_to_metadata(doc_name, collection_name, stats):
    """새 문서를 메타데이터에 추가"""
    metadata = load_document_metadata()
    metadata[collection_name] = {
        "name": doc_name,
        "collection_name": collection_name,
        "upload_date": datetime.now().isoformat(),
        "last_accessed": datetime.now().isoformat(),
        "statistics": stats
    }
    save_document_metadata(metadata)

def update_document_access_time(collection_name):
    """문서 접근 시간 업데이트"""
    metadata = load_document_metadata()
    if collection_name in metadata:
        metadata[collection_name]["last_accessed"] = datetime.now().isoformat()
        save_document_metadata(metadata)

def delete_document_from_metadata(collection_name):
    """메타데이터에서 문서 삭제"""
    metadata = load_document_metadata()
    if collection_name in metadata:
        del metadata[collection_name]
        save_document_metadata(metadata)

# Streamlit 설정
st.set_page_config(page_title="📄 다중 문서 RAG 챗봇", layout="wide")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "context_chat_mode" not in st.session_state:
    st.session_state.context_chat_mode = False
if "context_chunks" not in st.session_state:
    st.session_state.context_chunks = []
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = init_chromadb()
if "document_metadata" not in st.session_state:
    st.session_state.document_metadata = load_document_metadata()
if "selected_documents" not in st.session_state:
    st.session_state.selected_documents = []
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "전체 문서"

# 사이드바 - 문서 관리
with st.sidebar:
    st.markdown("## 📚 문서 관리")
    
    # Ollama 연결 상태
    if check_ollama_connection():
        st.success("🟢 Ollama 서버 연결됨")
    else:
        st.error("🔴 Ollama 서버 연결 안됨")
    
    # 기존 문서 목록
    existing_collections = [col.name for col in st.session_state.chroma_client.list_collections()]
    document_list = []
    
    for collection_name in existing_collections:
        if collection_name in st.session_state.document_metadata:
            doc_info = st.session_state.document_metadata[collection_name]
            document_list.append({
                "collection_name": collection_name,
                "display_name": doc_info["name"],
                "upload_date": doc_info["upload_date"],
                "statistics": doc_info["statistics"]
            })
    
    if document_list:
        st.markdown("### 📄 저장된 문서들")
        
        # 검색 모드 선택
        search_mode = st.selectbox(
            "🔍 검색 모드",
            ["단일 문서", "다중 문서", "전체 문서"],
            index=["단일 문서", "다중 문서", "전체 문서"].index(st.session_state.search_mode)
        )
        st.session_state.search_mode = search_mode
        
        # 문서 선택 인터페이스
        if search_mode == "단일 문서":
            selected_doc = st.selectbox(
                "문서 선택",
                options=[None] + [doc["collection_name"] for doc in document_list],
                format_func=lambda x: "선택하세요" if x is None else next(doc["display_name"] for doc in document_list if doc["collection_name"] == x),
                key="single_document_selector"
            )
            st.session_state.selected_documents = [selected_doc] if selected_doc else []
            
        elif search_mode == "다중 문서":
            st.markdown("검색할 문서들을 선택하세요:")
            selected_docs = []
            for doc in document_list:
                if st.checkbox(
                    doc["display_name"], 
                    key=f"multi_select_{doc['collection_name']}",
                    value=doc["collection_name"] in st.session_state.selected_documents
                ):
                    selected_docs.append(doc["collection_name"])
            st.session_state.selected_documents = selected_docs
            
        else:  # 전체 문서
            st.session_state.selected_documents = [doc["collection_name"] for doc in document_list]
            st.info(f"모든 문서 ({len(document_list)}개) 검색")
        
        # 선택된 문서 정보 표시
        if st.session_state.selected_documents:
            st.markdown("### 📊 선택된 문서 정보")
            total_paragraphs = 0
            total_pages = 0
            
            for collection_name in st.session_state.selected_documents:
                if collection_name in st.session_state.document_metadata:
                    doc_info = st.session_state.document_metadata[collection_name]
                    stats = doc_info["statistics"]
                    
                    with st.expander(f"📄 {doc_info['name']}", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("문단 수", stats["total_paragraphs"])
                            st.metric("페이지 수", stats["total_pages"])
                        with col2:
                            st.metric("기술 문단", f"{stats['technical_paragraphs']}")
                            st.metric("기술 비율", f"{stats['technical_ratio']:.1%}")
                        
                        # 문서 삭제 버튼
                        if st.button(f"🗑️ 삭제", key=f"delete_{collection_name}"):
                            if st.session_state.get(f"confirm_delete_{collection_name}", False):
                                # 실제 삭제 실행
                                try:
                                    st.session_state.chroma_client.delete_collection(collection_name)
                                    delete_document_from_metadata(collection_name)
                                    st.session_state.document_metadata = load_document_metadata()
                                    st.success(f"✅ {doc_info['name']} 삭제됨")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ 삭제 실패: {e}")
                            else:
                                st.session_state[f"confirm_delete_{collection_name}"] = True
                                st.warning("⚠️ 한 번 더 클릭하면 삭제됩니다!")
                                st.rerun()
                    
                    total_paragraphs += stats["total_paragraphs"]
                    total_pages += stats["total_pages"]
            
            # 전체 요약
            if len(st.session_state.selected_documents) > 1:
                st.markdown("### 📈 전체 요약")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 문서", len(st.session_state.selected_documents))
                with col2:
                    st.metric("총 문단", total_paragraphs)
                with col3:
                    st.metric("총 페이지", total_pages)
    
    else:
        st.info("아직 업로드된 문서가 없습니다.")
    
    st.markdown("---")
    
    # 고급 설정
    st.markdown("## ⚙️ 고급 설정")
    
    # 검색 방법 선택
    search_method = st.selectbox(
        "🔍 검색 방법",
        ["ChromaDB (기본)", "SentenceTransformer", "하이브리드"],
        help="ChromaDB: 빠름, SentenceTransformer: 정확함, 하이브리드: 둘 다 사용"
    )
    
    # 관련 문단 수 설정
    top_k = st.slider("📄 관련 문단 수", min_value=5, max_value=30, value=15)
    
    # 필터링 패턴 설정
    st.markdown("🧹 **필터링할 문구 패턴** (정규표현식)")
    default_patterns = "^Contents$\n^Page \\d+ of \\d+\n^Revision Tracking Summary"
    user_patterns = st.text_area("제외할 문단 패턴", value=default_patterns, height=100)

# 메인 영역
st.title("📄 다중 문서 RAG 챗봇")
st.markdown("여러 문서에서 동시 검색하여 질문에 답변하고, 선택한 내용으로 채팅을 이어갈 수 있어요.")

# PDF 업로드 섹션
uploaded_pdfs = st.file_uploader("📥 새 PDF 업로드", type=["pdf"],accept_multiple_files=True)

if uploaded_pdfs:
    for uploaded_pdf in uploaded_pdfs:          # 반복해서 처리
        pdf_name = uploaded_pdf.name
        collection_name = sanitize_collection_name(uploaded_pdf.name)
        
        # 이미 존재하는 문서인지 확인
        if collection_name not in [col.name for col in st.session_state.chroma_client.list_collections()]:
            with st.spinner("📄 문서 분석 및 임베딩 중..."):
                # 무의미한 문단 필터 생성
                is_meaningless = build_is_meaningless_checker(user_patterns)
                
                # 문단 추출 및 필터링
                before_doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
                before_count = sum(len(re.split(r'\n{2,}', p.get_text())) for p in before_doc)
                uploaded_pdf.seek(0)
                
                sentences_with_meta = extract_paragraphs_with_meta(uploaded_pdf, is_meaningless)
                after_count = len(sentences_with_meta)
                
                 # 👇 여기에서 문단 개수 체크!
                if after_count == 0:
                    st.error(f"❌ {pdf_name}에서 유효한 문단을 찾지 못했습니다. 필터링 규칙을 확인하거나 다른 PDF를 업로드하세요.")
                    continue  # 이 파일은 건너뜀

                # 통계 정보 생성
                doc_stats = get_document_statistics(sentences_with_meta)
                
                # ChromaDB에 저장
                collection = st.session_state.chroma_client.create_collection(name=collection_name)
                for idx, sentence in enumerate(sentences_with_meta):
                    for key, value in sentence.items():
                        if isinstance(value, list):
                            sentence[key] = ", ".join(map(str, value))
                        elif not isinstance(value, (str, int, float, bool)):
                            sentence[key] = str(value)
                    collection.add(
                        documents=[sentence["text"]],
                        metadatas=[sentence],
                        ids=[f"{collection_name}_{idx}"]
                    )
                
                # 메타데이터에 추가
                add_document_to_metadata(pdf_name, collection_name, doc_stats)
                st.session_state.document_metadata = load_document_metadata()
                
                # 새 문서를 자동으로 선택
                if collection_name not in st.session_state.selected_documents:
                    st.session_state.selected_documents.append(collection_name)
                
                st.success(f"✅ {after_count}개의 문단이 분석 및 저장되었습니다. (제외된 문단 수: {before_count - after_count})")
                st.rerun()
        else:
            st.info(f"✅ 이미 존재하는 문서: {pdf_name}")

# 검색 엔진 초기화
if st.session_state.selected_documents:
    if "search_engine" not in st.session_state or st.session_state.get("last_selected_docs") != st.session_state.selected_documents:
        st.session_state.search_engine = MultiDocumentSearchEngine(
            st.session_state.chroma_client, 
            st.session_state.selected_documents
        )
        st.session_state.last_selected_docs = st.session_state.selected_documents.copy()

# 채팅 모드 상태 표시
col1, col2 = st.columns([3, 1])

with col2:
    if st.session_state.context_chat_mode:
        st.info("💬 **컨텍스트 채팅 모드**\n선택된 문단을 기반으로 채팅 중")
        if st.button("🔄 전체 검색 모드로 돌아가기"):
            st.session_state.context_chat_mode = False
            st.session_state.context_chunks = []
            st.rerun()
    else:
        if st.session_state.selected_documents:
            if len(st.session_state.selected_documents) == 1:
                doc_name = st.session_state.document_metadata.get(st.session_state.selected_documents[0], {}).get("name", "문서")
                st.info(f"📖 **단일 문서 모드**\n{doc_name}")
            else:
                st.info(f"📚 **다중 문서 모드**\n{len(st.session_state.selected_documents)}개 문서에서 검색")
        else:
            st.warning("📭 **문서를 선택해주세요**")

with col1:
    # 채팅 기록 표시
    if st.session_state.selected_documents and "search_engine" in st.session_state:
        for i, chat in enumerate(st.session_state.chat_history):
            with st.chat_message(chat["role"]):
                st.write(chat["content"])
                
                # 사용자 질문에 대한 검색 결과 표시
                if chat["role"] == "user" and "search_results" in chat:
                    with st.expander(f"🔍 검색 결과 ({len(chat['search_results'])}개 문단)"):
                        for j, result in enumerate(chat["search_results"][:5]):  # 상위 5개만 표시
                            score = result.get("score", 0)
                            page = result.get("page", "?")
                            doc_name = result.get("document_name", "알 수 없음")
                            
                            st.markdown(f"**[{j+1}] {doc_name} p.{page}** (유사도: {score:.3f})")
                            st.markdown(f"📝 {result['text'][:200]}...")
                            
                            # 개별 문단으로 채팅 시작 버튼
                            if st.button(f"💬 이 문단으로 채팅", key=f"chat_{i}_{j}"):
                                st.session_state.context_chat_mode = True
                                st.session_state.context_chunks = [result]
                                st.rerun()
        
        # 질문 입력
        user_input = st.chat_input("질문을 입력하세요... (예: 이 구성품은 어떤 역할을 하나요?)")
        
        if user_input:
            # 사용자 질문 추가
            st.session_state.chat_history.append({
                "role": "user", 
                "content": user_input,
                "search_results": []
            })
            
            with st.spinner("🔍 관련 문단 검색 중..."):
                if st.session_state.context_chat_mode:
                    # 컨텍스트 채팅 모드: 선택된 문단만 사용
                    search_results = st.session_state.context_chunks
                else:
                    # 다중 문서 검색 모드
                    initial_results = st.session_state.search_engine.search(
                        query=user_input,
                        method=search_method.split()[0].lower(),
                        top_k=min(top_k * 2, 20)  # 더 많이 검색한 후 재순위화
                    )
                    
                    # 다중 문서 결과에 대해 LLM 기반 재순위화 적용
                    if len(st.session_state.selected_documents) > 1 and len(initial_results) > top_k:
                        search_results = rank_cross_document_results(user_input, initial_results, top_k)
                    else:
                        search_results = initial_results[:top_k]
                    
                    # 각 결과에 문서 이름 추가
                    for result in search_results:
                        doc_collection = result.get("source_collection", "")
                        if doc_collection in st.session_state.document_metadata:
                            result["document_name"] = st.session_state.document_metadata[doc_collection]["name"]
                        else:
                            result["document_name"] = doc_collection
                
                # 검색 결과를 채팅 기록에 저장
                st.session_state.chat_history[-1]["search_results"] = search_results
            
            with st.spinner("🤖 답변 생성 중..."):
                # 답변 생성
                answer = generate_answer_with_context(search_results, user_input)
                
                # 답변 추가
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })
            
            # 선택된 문서들의 접근 시간 업데이트
            for collection_name in st.session_state.selected_documents:
                update_document_access_time(collection_name)
            
            st.rerun()
    
    else:
        st.info("📂 사이드바에서 문서를 선택하거나 새 PDF를 업로드해주세요.")

# 푸터
st.markdown("---")
st.markdown("💡 **사용 팁**: 여러 문서에서 동시 검색하고, 검색 결과에서 특정 문단을 선택하면 해당 문단을 중심으로 대화를 이어갈 수 있습니다!")
