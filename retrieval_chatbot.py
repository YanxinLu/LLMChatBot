from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory

# retriever usage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import re


class LangchainChatbot:
    def __init__(self,
                 api_key,
                 template_file_path='template.txt',
                 model_name_or_path='gpt-3.5-turbo'):
        self.template = open(template_file_path, 'r').read()
        self.model_name_or_path = model_name_or_path
        self.api_key = api_key
        self.model = self.get_model()
        self.retriever = None
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"],
            template=self.template
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.llm_chain = LLMChain(llm=self.model,
                                  prompt=self.prompt,
                                  verbose=True,
                                  memory=self.memory
                                  )

    def set_retriever_url(self, url):
        loader = WebBaseLoader(url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        self.retriever = vectorstore.as_retriever(k=4)

    def get_model(self):
        model_name_or_path = self.model_name_or_path
        api_key = self.api_key
        if 'gpt' in model_name_or_path:
            model = ChatOpenAI(model=model_name_or_path,
                               openai_api_key=api_key)
        elif 'claude' in model_name_or_path:
            model = ChatAnthropic(model=model_name_or_path,
                                  anthropic_api_key=api_key)
        elif 'gemini' in model_name_or_path:
            model = ChatGoogleGenerativeAI(model=model_name_or_path,
                                           google_api_key=api_key)
        else:
            llm = HuggingFaceHub(repo_id=model_name_or_path,
                                 huggingfacehub_api_token=api_key)
            model = ChatHuggingFace(llm=llm)
        return model

    def chat_with_chatbot(self, human_input):
        if self.retriever is not None:
            retriever_search = self.retrieve_by_retriever(human_input)
            human_input = f'{human_input}\nWeb Retriever: {retriever_search}'
        return self.llm_chain.predict(human_input=human_input)

    def retrieve_by_retriever(self, query):
        return '\n'.join(re.sub('\n+', '\n', dict(result)['page_content']) for result in self.retriever.invoke(query))

    def retrieve_by_memory(self, keyword):
        return [msg.content for msg in self.memory.chat_memory.messages if keyword in msg.content]


if __name__ == "__main__":
    model = 'gemini-pro'
    Chatbot = LangchainChatbot(api_key='your-api-key',
                               model_name_or_path=model)
    Chatbot.set_retriever_url("https://optimalscale.github.io/LMFlow/")
    Chatbot.chat_with_chatbot("What is LMFlow?")
    Chatbot.chat_with_chatbot("How can I fine-tune models using LMFlow?")
    with open(f"chat_result/{model}.txt", 'w') as file:
        file.write(str(Chatbot.memory.chat_memory))
