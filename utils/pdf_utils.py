import fitz  # PyMuPDF
import re
from typing import List, Dict, Callable, Optional

def extract_paragraphs_with_meta(pdf_file, is_meaningless_checker: Optional[Callable] = None) -> List[Dict]:
    """
    PDF에서 문단 단위로 텍스트를 추출하고, 페이지와 문단 인덱스 포함
    
    개선사항:
    - 더 정확한 문단 분할
    - 표와 이미지 캡션 처리
    - 메타데이터 풍부화
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    results = []
    total_paragraph_count = 0

    for page_number, page in enumerate(doc, start=1):
        page_text = page.get_text()
        
        # 더 정교한 문단 분할
        # 1. 기본 이중 개행 분할
        raw_paragraphs = re.split(r'\n{2,}', page_text)
        
        # 2. 추가 분할 패턴들
        refined_paragraphs = []
        for para in raw_paragraphs:
            # 긴 문단을 문장 단위로 추가 분할 (선택적)
            if len(para) > 1000:  # 1000자 이상인 경우
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z가-힣])', para)
                for sentence in sentences:
                    if len(sentence.strip()) > 30:
                        refined_paragraphs.append(sentence.strip())
            else:
                refined_paragraphs.append(para)
        
        # 3. 문단 정제 및 필터링
        paragraphs = []
        for para in refined_paragraphs:
            cleaned_para = clean_paragraph(para)
            if is_valid_paragraph(cleaned_para):
                paragraphs.append(cleaned_para)

        # 4. 메타데이터와 함께 저장
        for idx, paragraph in enumerate(paragraphs):
            # 무의미한 문단 체크
            if is_meaningless_checker and is_meaningless_checker(paragraph):
                continue
            
            # 문단 유형 분석
            para_type = analyze_paragraph_type(paragraph)
            
            # 키워드 추출
            keywords = extract_keywords(paragraph)
            
            metadata = {
                "text": paragraph,
                "page": page_number,
                "paragraph_index": idx,
                "type": para_type,
                "keywords": ", ".join(keywords) if isinstance(keywords, list) else str(keywords),

            }

            results.append(metadata)
            
            total_paragraph_count += 1

    doc.close()
    return results

def clean_paragraph(text: str) -> str:
    """
    문단 텍스트 정제
    """
    if not text:
        return ""
    
    # 1. 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 2. 하이픈으로 분할된 단어 복원
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # 3. 특수 문자 정리 (선택적)
    text = re.sub(r'[^\w\s\.\,\!\?\(\)\-\:\/\%\+\=\[\]\'\"가-힣]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_valid_paragraph(text: str, min_length: int = 50) -> bool:
    """
    유효한 문단인지 판단
    """
    if not text or len(text) < min_length:
        return False
    
    # 단순 반복 패턴 제거
    if len(set(text.split())) < 5:  # 고유 단어가 5개 미만
        return False
    
    # 숫자만 있는 문단 제거
    if re.match(r'^[\d\s\.\-]+', text):
        return False
    
    return True

def analyze_paragraph_type(text: str) -> str:
    """
    문단 유형 분석
    """
    text_lower = text.lower()
    
    # 제목 패턴
    if re.match(r'^\d+(\.\d+)*\s+[A-Z가-힣]', text) or len(text) < 100:
        return "heading"
    
    # 목록 패턴
    if re.match(r'^[-•◦▪▫]\s', text) or re.match(r'^\([a-z0-9]\)', text):
        return "list_item"
    
    # 표 관련
    if 'table' in text_lower or '표' in text:
        return "table_related"
    
    # 그림/도표 관련
    if any(word in text_lower for word in ['figure', 'fig.', '그림', '도']):
        return "figure_related"
    
    # 기술 사양
    if re.search(r'\d+\s*(mm|cm|m|kg|℃|°C|bar|psi|rpm)', text):
        return "specification"
    
    # 일반 텍스트
    return "content"

def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """
    문단에서 주요 키워드 추출
    """
    # 기술 용어 패턴
    technical_patterns = [
        r'\b[A-Z]{2,}\b',  # 약어 (예: API, ASME)
        r'\b\w*valve\w*\b',  # 밸브 관련
        r'\b\w*pump\w*\b',   # 펌프 관련
        r'\b\w*pipe\w*\b',   # 파이프 관련
        r'\b\w*pressure\w*\b',  # 압력 관련
        r'\b\w*temperature\w*\b',  # 온도 관련
    ]
    
    keywords = []
    text_lower = text.lower()
    
    for pattern in technical_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        keywords.extend([m for m in matches if len(m) > 2])
    
    # 한국어 기술 용어
    korean_terms = [
        '밸브', '펌프', '배관', '압력', '온도', '유량', '설비', '시스템',
        '제어', '안전', '검사', '시험', '기준', '규격', '재료', '구조'
    ]
    
    for term in korean_terms:
        if term in text:
            keywords.append(term)
    
    # 중복 제거 및 길이 제한
    unique_keywords = list(set(keywords))[:max_keywords]
    return unique_keywords

def has_technical_terms(text: str) -> bool:
    """
    기술 용어 포함 여부 확인
    """
    technical_indicators = [
        r'\d+\s*(mm|cm|m|kg|℃|°C|bar|psi|rpm|mpa)',  # 단위
        r'\b(pressure|temperature|flow|valve|pump|pipe)\b',  # 영어 기술용어
        r'(압력|온도|유량|밸브|펌프|배관|설비|시스템)',  # 한국어 기술용어
        r'\b[A-Z]{2,}\b',  # 약어
        r'(API|ASME|ANSI|ISO|KS)\s*\d+',  # 표준 규격
    ]
    
    for pattern in technical_indicators:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False

def build_is_meaningless_checker(user_input: str) -> Callable[[str], bool]:
    """
    사용자가 입력한 정규표현식들을 기반으로 필터링 함수 생성
    
    개선사항:
    - 기본 필터 패턴 추가
    - 에러 처리 강화
    """
    # 사용자 패턴 파싱
    user_patterns = [line.strip() for line in user_input.strip().split("\n") if line.strip()]
    
    # 기본 필터 패턴들
    default_patterns = [
        r'^Page\s+\d+\s+of\s+\d+',
        r'^Contents$',
        r'^Cover\s*Page$',
        r'^Revision Tracking Summary$',
        r'^\s*$',
        r'^\s*[_\-=]{3,}\s*$',
        r'^(Figure|Fig\.)\s*\d+',
        r'^(Table|표)\s*\d+',
        r'©.*\d{4}',
    ]
    
    all_patterns = user_patterns + default_patterns
    
    def checker(text: str) -> bool:
        """실제 필터링 함수"""
        if not text or len(text.strip()) < 10:  # 너무 짧은 텍스트
            return True
        
        text_stripped = text.strip()
        for pattern in all_patterns:
            try:
                if re.search(pattern, text_stripped, flags=re.IGNORECASE | re.MULTILINE):
                    return True
            except re.error as e:
                print(f"정규표현식 오류 (패턴: {pattern}): {e}")
                continue
        return False
    return checker

def get_document_statistics(sentences_with_meta: List[Dict]) -> Dict:
    """
    문서 통계 정보 생성
    """
    if not sentences_with_meta:
        return {}
    
    total_paragraphs = len(sentences_with_meta)
    total_pages = max(s["page"] for s in sentences_with_meta)
    total_chars = sum(s["char_count"] for s in sentences_with_meta)
    total_words = sum(s["word_count"] for s in sentences_with_meta)
    
    # 문단 유형별 통계
    type_counts = {}
    for s in sentences_with_meta:
        para_type = s.get("paragraph_type", "unknown")
        type_counts[para_type] = type_counts.get(para_type, 0) + 1
    
    # 기술 문단 비율
    technical_count = sum(1 for s in sentences_with_meta if s.get("has_technical_terms", False))
    
    return {
        "total_paragraphs": total_paragraphs,
        "total_pages": total_pages,
        "total_characters": total_chars,
        "total_words": total_words,
        "avg_chars_per_paragraph": total_chars // total_paragraphs if total_paragraphs > 0 else 0,
        "avg_words_per_paragraph": total_words // total_paragraphs if total_paragraphs > 0 else 0,
        "paragraph_types": type_counts,
        "technical_paragraphs": technical_count,
        "technical_ratio": technical_count / total_paragraphs if total_paragraphs > 0 else 0
    }
