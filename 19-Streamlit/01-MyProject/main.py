import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from dotenv import load_dotenv
from langchain import hub

load_dotenv()

st.title("나만의 AI 테스트")

# 처음 한 번만 실행
if "messages" not in st.session_state:
    # 대화기록 저장
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")
    option = st.selectbox(
        "프롬프트를 선택해 주세요", ("기본모드", "SNS 게시글", "요약"), index=0
    )
# 이전 대화를 출력
# for role, message in st.session_state["messages"]:
#     st.chat_message(role).write(message)
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)
        # st.write(f"{chat_message.role}: {chat_message.content}")

# 새로운 메지 추가    
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

def create_chain():
    # prompt | llm | output_parser
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절한 AI 어시스턴스입니다."),
            ("user", "#Question:\n{question}"),
        ]
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    output_parser = StrOutputParser()

    # 체인 생성
    chain = prompt | llm | output_parser
    return chain

# 초기화 버튼이 눌리면
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어 보세요!")

if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)
    # chain 생성
    chain = create_chain()
    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        # 반 공간(컨테이너)을 만들어서 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # ai_answer = chain.invoke({"question": user_input})
    # AI 답변
    # st.chat_message("assistant").write(ai_answer)

    # 대화 기록 저장
    # st.session_state["messages"].append(("user", user_input))
    # st.session_state["messages"].append(("assistant", user_input))

    add_message("user", user_input)
    add_message("assistant", ai_answer)

