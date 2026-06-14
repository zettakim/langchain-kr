import streamlit as st
from langchain_core.messages.chat import ChatMessage 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_openai import ChatOpenAI
from langchain import hub
from dotenv import load_dotenv
import glob

st.title("나만의 챗GPT💬")

# 대화 기록을 저장하기 위한 용도로 생성한다.
# dict 구조로 저장
# 페이지 전체가 새로고침되어도 저장됨

# 처음 한 번 실행 필요, 이후에는 초기화 안됨
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 초기하 버튼 생성
    clear_btn = st.button("대화 초기화")

    prompt_files = glob.glob("prompts/*.yaml")
    selected_prompt = st.selectbox("프롴프트를 선택해 주세요", prompt_files, index=0)
    task_input = st.text_input("TASK 입력", "")

    # selected_prompt = st.selectbox(
    #     "프롬프트를 선택해 주세요", ("기본모드", "SNS 게시글", "요약"), index=0
    # )


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메세지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 체인 생성
def create_chain(prompt_filepath, task=""): 
    # prompt | llm | oitput_parser

    # Prompt 적용
    prompt = load_prompt(prompt_filepath, encoding="utf-8")
    if task:
        prompt = prompt.partial(task=task)

    # prompt(기본모드)
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "당신은 친정한 AI 어시스넡트입니다. 다음의 질문에 간결하게 답변 해 주세요."),
    #         ("user", "#Question:\n{question}"),
    #     ]
    # )
    
    # 프롴프트 적용
    # prompt = load_prompt(prompt_filepath, encoding="utf-8")

    # if prompt_type == "SNS 게시글":
    #     prompt = load_prompt("prompts/sns.yaml", encoding="utf-8")
    # elif prompt_type == "요약":
    #     # 요약 프롴프트
    #     prompt = load_prompt("prompts/summary.yaml", encoding="utf-8")
    #     # prompt = hub.pull("teddynote/chain-of-density-map-korean")
 
    # llm
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    chain = prompt | llm | output_parser
    return chain

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 출력
print_messages()

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")
 
# 만약에 사용자 입력이 들어오면...
if user_input:
    # 사용자 입력
    st.chat_message("user").write(user_input)
    
    # chain 생성
    chain = create_chain(selected_prompt, task=task_input)

    # 스트리밍 호출
    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토근을 스트리밍 출력한다.
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)
    
    # ai_answer = chain.invoke({"question": user_input})

    # ai 답변
    # st.chat_message("assistant").write(ai_answer)

    # st.chat_message("user").write(user_input)
    # 동일
    # with st.chat_message("user"):
    #     st.write(user_input)

    # 대화기록을 저장한다
    add_message("user", user_input)
    add_message("assistant", ai_answer)

