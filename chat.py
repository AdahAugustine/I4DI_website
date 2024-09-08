import os 
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_community.document_loaders import (CSVLoader,
#                                       EverNoteLoader,
#                                       PyMuPDFLoader,
#                                       TextLoader,
#                                       UnstructuredEmailLoader,
#                                       UnstructuredHTMLLoader,
#                                       UnstructuredMarkdownLoader,
#                                       UnstructuredEPubLoader,
#                                       UnstructuredODTLoader,
#                                       UnstructuredPowerPointLoader,
#                                       UnstructuredWordDocumentLoader,
#                                       UnstructuredExcelLoader,
#                                     UnstructuredImageLoader,
#                                     SRTLoader,
#                                         WebBaseLoader)
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# import pandas as pd
from dotenv import load_dotenv, find_dotenv
import re
from langchain_groq import ChatGroq

# os.environ["HF_TOKEN"]="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"
# os.environ["GROQ_API_KEY"]="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"
# g_llm=ChatGroq(model="llama3-70b-8192", temperature=0)
# os.environ["OPENAI_API_KEY"]="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"

load_dotenv(find_dotenv())

llm=ChatOpenAI(temperature=0)

Service_prompts =PromptTemplate(
    template="""Services, in terms of work, refer to the tasks or activities performed by individuals or organizations to meet specific needs.
    I4DI provides some specific services to private and international organisation. The broad services are Advisory services, Tech-enabled Knowledge Products and Tech System Solutions, 
    provided in the fields of sustainability and climate, MERL, Urban Development and Impact Technology.
    Output Just "Yes" if the question asks about information about services rendered in general or these aforementioned specific services  Or Output "No" if the question does not ask about information about services rendered in general or these aforementioned specific services, without any preamble or explanation.
    Here is the input question: {question}""",
     input_variables=["question"])

Capabilities_prompts =PromptTemplate(
    template="""Corporate capabilities, also known as organizational capabilities, are fundamental building blocks for a company. They form the basis of a firm’s competitive advantage and are essential for achieving strategic goals.
    I4DI's capabilities are Evaluation Capabilities, WASH Capabilities,Knowledge Management Capability,Africa Capability,Advanced data analytics Capabilities,Zen-o Capabilities.
    Output Just "Yes" if the question asks about information about the continent of Africa, WASH, Capabilities of I4DI, Data analysis work, Zen-O and data analytics capabilities or Output just "No" if the question does not ask about information about the continent of Africa, Project details, WASH, Capabilities of I4DI, Data analysis work, Zen-O and statistic achievements like number of this and that and how many, without any preamble or explanation

     Here is the input question: {question}""",
     input_variables=["question"])

Focus_Area_prompts =PromptTemplate(
    template="""Focus areas are the foundation stones of strategy. They expand on a companies Vision Statement and provide structure for achieving organizational goals.
    I4DI focus areas are Sustainability and Climate ,Monitoring, Evaluation, Research, and Learning, Urban Development and Impact Technology. 
    
    Output Just "Yes" if the question asks about these specific focus areas, people in charge of focus areas in I4DI or focus area as a subject in general or Just"No" if the question does not ask about these specific focus areas, people in charge of focus areas in I4DI or focus area as a subject in general, without any preamble or explanation.
    Here is the input question: {question}""",
    input_variables=["question"])


Publication_Prompt=PromptTemplate(
    template="""I4DI has a lot of published articles.
    Output Just "Yes" if the question asks about information related to a study or a project, not the number of projects, but the discussion of a project or a published article or Just "No" if the question does not asks about information related to a study or a project, without any preamble or explanation.
     Here is the input question: {question}""",
     input_variables=["question"])


Project_Prompt=PromptTemplate(
    template=""" I4DI has conducted a lot of projects.
     Output Just "Yes" if the question asks about projects conducted and not number of projects conducted information about this theme Or "No" if the question does not ask about information about this theme, without any preamble or explanation
     Here is the input question: {question}""",
     input_variables=["question"])


Number_Prompt=PromptTemplate(
    template=""" I4DI has had a lot of achievements, results and accomplishments. Results includes statistical information such as, 
    how many projects, how many dashboards, people trained, what projects were conducted in a region, state, continent, project details e.t.c
    Output Just "Yes" if the question asks about information about I4DI's results Or "No" if the question does not ask about information about I4DI's results, without any preamble or explanation
     Here is the input question: {question}""",
     input_variables=["question"])


News_Product=PromptTemplate(
    template=""" I4DI has had a number of events occuring to them with history of said events, such as time of awarded contract, time when an agreement was made with another organisation .e.t.c
    Output Just "Yes" if the question asks about news and insights related to I4DI Or "No" if the question does not ask about news and insights related to I4DI, without any preamble or explanation
     Here is the input question: {question}""",
     input_variables=["question"])


Products_Prompt=PromptTemplate(
    template=""" I4DI products are Zen-O, Zen-O Project, Zen-O Consultants, Zen-O Learn, Data Visualization, Jargonator
    Output Just "Yes" if the question asks inquires about Zen-O or any of I4DI's product Or "No" if the question does not ask inquires about Zen-O or any of I4DI's product, without any preamble or explanation
     Here is the input question: {question}""",
     input_variables=["question"])


About_Us_Prompt=PromptTemplate(
    template=""" Just like any other company, I4DI has mission, vision, a story, a history, an edge over other companies, Careers,Job Opportunity, Roster, Location and Contact and has had contract awarded.
    Output Just "Yes" if the question asks about I4DI's mission, vision, story, history, edge over other companies, Careers,Job Opportunity, Roster, Location and Contact, contract awarded Or "No" if the question does not ask about I4DI's mission, vision, story, history, edge over other companies, Careers,Job Opportunity, Roster, Location and Contact, contract awarded
     Here is the input question: {question}""",
     input_variables=["question"])


Employees=PromptTemplate(
    template="""This is a list of I4DI's employees with their job titles, seperated by a comma
                Franz Valli,  Humanitarian Assistance and Development Advisor
                Molly Hageboeck,  MERL Director Emerita
                Michael Buret,  Regional Director, Middle East & North Africa
                Zehra K. Dzihic,  Senior Program Development and Effectiveness Expert Advisor
                Sarah Dawn Petrin,  Conflict & Displacement Consultant
                Palak Agarwal,  Data Scientist
                Abi Riak,  Senior Organizational Effectiveness Advisor
                Lauren Ropp,  Associate Director for Program Management
                Steven Lichty, Evaluation & Research Advisor
                Azra K. Nurkic, CEO and Co founder
                Christine Traylor, Director of Operations
                Jovana Bulatovic,  Evaluation & Research Associate
                Troy Wray,  Chief Strategy and Growth Officer
                Adi Karisik,  Senior Advisor for Global Cybersecurity and Operational Technology
                Sandra Moscoso,  Senior Performance & Knowledge Management Advisor
                Jon-Paul Bowles,  Sustainable Development Advisor
                Olivier Payen,  Economic Development & Education Advisor
                Ochanya Adah,  Technology Design & Program Specialist
                Adnan Hadrovic,  Senior Global Security & Diplomacy Advisor
                Emir Nurkic Kacapor, Chief Operations Officer & Co-Founder"
                Dzenana Sabic Hamidovic,  Social Development Advisor
                Kemal Sokolović,  Chief Data and Technology Officer
                Amel Osmanovic,  Full Stack Web Developer
                Vahid Rahic,  Mobile App Developer
                Saurav Behera,  Finance and Admin Manager
                Denis Drekovic,  Digital Tech & Data Science Team Lead
                Brandan Hamlin,  Data For Social Change Intern
                Andrea Pozderac,  Analyst
                Jigyasa Sidana,  Program Management and MERL Associate
                Beth Sorel,  Program Management and MERL Associate
                Jdhymi Dulaurier,  Associate Director of Program Management
                Sam Bolte,  Special Initiatives Analyst
                Dan Robinson,  Director for MERL
                Grace Vottero,  Associate Director for Program Management
                Penelope Kogan,  Analyst
                Amina Jusupovic,  Mobile App Developer
    Output Just "Yes" if the question asks about any of the employees or roles, including their responsibilities or Just "No" if the question does not ask about any of the employees or roles, without any preamble or explanation
    Here is the input question: {question}""",
    input_variables=["question"])


Prompts=[Service_prompts, Capabilities_prompts,Focus_Area_prompts,Publication_Prompt,Project_Prompt,Number_Prompt,News_Product,Products_Prompt,About_Us_Prompt,Employees]

embeddings=OpenAIEmbeddings()
docsearch=Chroma(persist_directory="./web_embeddings",embedding_function=embeddings)
Service_embedddings=Chroma(persist_directory="./service",embedding_function=embeddings)
Cap_embedddings=Chroma(persist_directory="./capabil",embedding_function=embeddings)
Focus_embedddings=Chroma(persist_directory="./Focus",embedding_function=embeddings)
Pub_embedddings=Chroma(persist_directory="./pubs",embedding_function=embeddings)
Project_embedddings=Chroma(persist_directory="./Project",embedding_function=embeddings)
Number_embedddings=Chroma(persist_directory="./Number",embedding_function=embeddings)
News_embedddings=Chroma(persist_directory="./News",embedding_function=embeddings)
Product_embedddings=Chroma(persist_directory="./Product",embedding_function=embeddings)
About_embedddings=Chroma(persist_directory="./About",embedding_function=embeddings)
Employee_embedddings=Chroma(persist_directory="./Employee",embedding_function=embeddings)


Embeddings=[Service_embedddings,Cap_embedddings,Focus_embedddings,Pub_embedddings,
            Project_embedddings,Number_embedddings,News_embedddings,
            Product_embedddings,About_embedddings,Employee_embedddings ]


df=pd.read_csv("new_df.csv")

# df.columns

llm_code= ChatOpenAI(temperature=0, model_name="gpt-4")
llm_context=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationKGMemory, CombinedMemory

df=df.drop(["country_temp","url"], axis="columns")

chat_history_buffer=ConversationBufferWindowMemory(k=5,
                                                     memory_key="chat_history_buffer",
                                                     input_key="input"
                                                    )
chat_history_summary = ConversationSummaryMemory(llm=llm_context,
                                                memory_key="chat_history_summary",
                                                input_key="input")
chat_history_KG= ConversationKGMemory(llm=llm_context,
                                     memory_key="chat_history_KG",
                                     input_key="input")
memory= CombinedMemory(memories=[chat_history_buffer,chat_history_summary,chat_history_KG])

PREFIX="""You are working with a pandas dataframe in Python. The name of the dataframe is`df`.You should use the tools below to answer the question posed of you:
Summary of the whole conversation:
{chat_history_summary}

Last few messages between you and user:
{chat_history_buffer}

Entities that the conversation is about:
{chat_history_KG}

If there is a url link, output the Url link.
Final output answer should be detailed and elaborate.

"""

agent=create_pandas_dataframe_agent(llm_code,
                                    df,
                                    prefix=PREFIX,
                                    verbose=True,
                                    agent_executor_kwargs={"memory":memory},
                                    allow_dangerous_code=True,
                                    input_variables=["df_head","input","agent_scratchpad","chat_history_buffer","chat_history_summary","chat_history_KG"]
                                   )

def tag_question(question):
    tags=[]
    for i in Prompts:
        chain= i| llm | StrOutputParser()
        tags.append(chain.invoke({"question":question}))
    return tags
    
## Workflow Prompts

rag_prompt=PromptTemplate(
    template= """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an assistant for question-answering tasks employed by I4di to answer questions regarding their website information.
    Use the following pieces of retrieved context to answer the question.
    You write in natural language at a 10 grade reading level
    Always use the document context to create your responses.
    If you don't know the answer, just say that you don't know.
    Answer just the question asked and be detailed in the answer.
    If a link exist in the context, always output the most important links as part of the answer.
    Try to output up to 4 or more links
    This is so as to tell the user to get more information from the link.
    Do not start the answer as if you are answering from the context but as if you are answering the question in charge of I4DI. 
    For example instead of "Based on information provided, the CEO is ...", you answer in the form of "The CEO is ..."
    Do not refer to I4DI as "they" or "their" but as "we", "us" and "our"
     <|eot_id|><|start_header_id|>user<|end_user_id|>
    Question:{question}
    Context:{context}
    Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"])
# format the document and src
def format_con(docs):
  return " ".join(doc.page_content for doc in docs)
def format_src(docs):
  return "\n\n".join(doc.metadata["source"].split("/")[-1] for doc in docs)
rag_chain=rag_prompt | llm | StrOutputParser()


rewrite_prompt=PromptTemplate(template=
                              """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You a question re-writer
                              that converts an input question to a better version that is optimized \n
                              for better vector database search for I4DI. Rewrite the question to specifically ask about I4DI. 
                              For example, transform 'Who is Amel' to 'Who is Amel in I4DI'. Ensure the rewritten question retains 
                              the original intent while focusing on I4DI. 
                              Output the question as a string without any preamble or explanation
                              <|eot_id|><|start_header_id|>user<|end_user_id|>
                              Here is the input question:{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                              input_variables=["question"])
rewrite_chain=rewrite_prompt| llm| StrOutputParser()

base_prompt=PromptTemplate(template=
                              """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You a question re-writer
                              that converts an input question to a base version that is optimized \n
                              for better web or vector database search for I4DI. Rewrite the question to its base form related to I4DI. 
                              For example, transform 'Who is the CEO' to 'CEO in I4DI'. transform  
                              Ensure the rewritten question simplifies the query while focusing on roles or entities within I4DI 
                              Output the question as a string without any preamble or explanation
                              <|eot_id|><|start_header_id|>user<|end_user_id|>
                              Here is the input question:{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                              input_variables=["question"])
base_chain=base_prompt| llm| StrOutputParser()

agent_question=PromptTemplate(template=
                              """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a question re-writer that 
                              transforms specific input questions into more general, simplified questions optimized for a pandas agent.
                              For instance, change "Who is the CEO in I4DI" to "Who is the CEO." 
                              If the term "I4DI" appears in the input question, remove it and rewrite the question. 
                              For instance, change "How many projects has I4DI conducted in Asia?" to "How many projects has been conducted in Asia"
                              Output the rewritten question as a string without any preamble or explanation.
                              <|eot_id|><|start_header_id|>user<|end_user_id|>
                              Here is the input question:{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                              input_variables=["question"])
agent_chain=agent_question| llm| StrOutputParser()

cojoin_question=PromptTemplate(template=
                              """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Given a chat history and the latest user
                              question which might reference context in the chat history, formulate a standalone question which can be understood
                              without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is
                              <|eot_id|><|start_header_id|>user<|end_user_id|>
                              chat history:{chat_history},
                              user question:{user_question}
                              <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                              input_variables=["chat_history","user_question"])
cojoin_chain=cojoin_question| llm| StrOutputParser()

r_grader_prompt=PromptTemplate(template= """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance of a retrieved document
                                    to a user question. If the document contains keywords related to the user question,grade it as relevant.It does not need to be a
                                    stringent test,the goal is to filter out erroneous retrievals. \n
                                    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                                    Provide the binary score output as just a string of 'yes' and 'no' with no preamble or explanation
                                    <|eot_id|><|start_header_id|>user<|end_user_id|>
                                    Here is the retrieved document:\n\n {document} \n\n
                                    Here is the user question:{question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                    """,
                                    input_variables=["question", "document"])
retrieval_grader=r_grader_prompt | llm | StrOutputParser()

# Hallucination grader: Grade the answers against the context. This is to check if the model is trying to get answers outside the context
hallucination_prompt=PromptTemplate(template= """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
                                    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
                                    whether the answer is grounded in supported by a set of facts. Provide the binary score
                                     output as just a string of 'yes' and 'no' with no preamble or explanation
                                    <|eot_id|><|start_header_id|>user<|end_user_id|>
                                    Here are the facts:
                                    \n----------\n
                                    {document}
                                    \n----------\n
                                    Here is the answer:{generation} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                                    input_variables=["generation", "document"])
hallucination_grader=hallucination_prompt| llm | StrOutputParser()

# answer_grader: Grade usefulness of the answer to the question
answer_g_prompt=PromptTemplate(template= """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
                                    an answer is useful to resolve a question. Give a binary 'yes' or 'no' score to indicate
                                    whether the answer is useful to resove a question.Provide the binary score
                                     output as just a string of 'yes' and 'no' with no preamble or explanation
                                    <|eot_id|><|start_header_id|>user<|end_user_id|>
                                    Here is the answer:
                                    \n----------\n
                                    {generation}
                                    \n----------\n
                                    Here is the question:{question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                                    input_variables=["generation", "question"])
answer_grader=answer_g_prompt| llm| StrOutputParser()

greet_prompt=PromptTemplate(template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Check the question for any form of greeting such as:

"Hello"
"Hi"
"Hey"
"How are you?"
"How was your day?"
"What's up?"
"How's it going?"
"How is your day going?"
"How is your chatbot doing?"
Return "Yes" if found; otherwise, return "No" without any explanations or preamble.
Here is the question:{question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                                    input_variables=['question'])
greet=greet_prompt| llm| StrOutputParser()

greet_response_prompt=PromptTemplate(template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Respond to the user's greeting or question by informing them that you are a virtual assistant employed by I4DI, 
ready to help with any questions, concerns, or partnership requests related to I4DI. 
Also, provide the URL "https://i4di.org" for more information about I4DI.
Here is the question:{question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                                    input_variables=['question'])
greet_r=greet_response_prompt| llm| StrOutputParser()

answer_prompt=PromptTemplate(template= """<|begin_of_text|><|start_header_id|>system<|end_header_id|> Two answers exist for one question,
                                        if the input question inquires more about number of project, consider {generation_1} answer 
                                        to be more important to answering the question.
                                     output as just an answer string with no preamble or explanation
                                    <|eot_id|><|start_header_id|>user<|end_user_id|>
                                    Here is the first answer:
                                    \n----------\n
                                    {generation_1}
                                    \n----------\n
                                    Here is the second answer:
                                    \n----------\n
                                    {generation_2}
                                    \n----------\n
                                    Here is the question:{question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                                    input_variables=["generation_1","generation_2", "question"])
answer=answer_prompt| llm| StrOutputParser()


def format_con(docs):
  return " ".join(doc.page_content for doc in docs)
def format_src(docs):
  return "\n\n".join(doc.metadata["source"].split("/")[-1] for doc in docs)


def format_content_src(docs):
    url_links=[]
    for doc in docs:
        url=re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+',doc.page_content)
        links=[]
        for urls in url:
            if urls.startswith("https://")== True:
                links.append(urls)
        for link in links:
            url_links.append(link)
    return url_links

def query_from_tag(question,k):
    global div
    div=tag_question(question)
    retrieved_doc=[]
    general="Yes" in div
    
    for i,j in zip(div,Embeddings):
        if "No" in i.split(" "):
            pass
        elif j== Number_embedddings:
            try:
                agent_question=agent_chain.invoke({"question":question})
                ans=agent.invoke(agent_question)
                agent_ans=ans["output"]
            except:
                pass
            retriever=MultiQueryRetriever.from_llm(retriever=j.as_retriever(search_type='mmr',search_kwargs={'k':k, 'fetch_k':10}), llm=llm)
            # retriever=j.as_retriever(search_type='mmr',search_kwargs={'k':k, 'fetch_k':10})
            docs=retriever.invoke(question)
            for doc in docs:
                if doc not in retrieved_doc:
                    retrieved_doc.append(doc)
        # elif retrieved_doc != []:
        else:
            retriever=MultiQueryRetriever.from_llm(retriever=j.as_retriever(search_type='mmr',search_kwargs={'k':k, 'fetch_k':10}), llm=llm)
            # retriever=j.as_retriever(search_type='mmr',search_kwargs={'k':k, 'fetch_k':10})
            docs=retriever.invoke(question)
            for doc in docs:
                if doc not in retrieved_doc:
                    retrieved_doc.append(doc)
    if retrieved_doc==[]:
        retriever=MultiQueryRetriever.from_llm(retriever=docsearch.as_retriever(search_type='mmr',search_kwargs={'k':k, 'fetch_k':10}), llm=llm)
        # retriever=docsearch.as_retriever(search_type='mmr',search_kwargs={'k':k, 'fetch_k':10})
        docs=retriever.invoke(question)
        for doc in docs:
            if doc not in retrieved_doc:
                retrieved_doc.append(doc)
    try:
        return retrieved_doc,agent_ans
    except:
        return retrieved_doc,"Nil"

chat_history=[]
# def workflow(question, history):
def get_response(msg):
    greetings=greet.invoke({"question":msg})
    if "Yes" in greetings:
        response=greet_r.invoke({"question":msg})
        # history.append((question,response))
        # return "", history, ""
        return response
    else:
        documents=[]
        init_question=msg
        # question=rewrite_chain.invoke(question)
        cojoined=cojoin_chain.invoke({"chat_history":chat_history, "user_question":msg})
        # print(cojoined)
        base_q=base_chain.invoke(cojoined)
        b_docs,agent_ans=query_from_tag(base_q,3)
        for i in b_docs:
            if i in documents:
                pass
            else:
                documents.append(i)
        # for i,j in enumerate(documents):
        #     r_temp=documents[i]
        #     r_grade=retrieval_grader.invoke({"question":cojoined, "document":r_temp.page_content})
        #     if r_grade=='no':
        #         documents.remove(r_temp)
        print(len(documents))
        chat_history.append(cojoined)
        if len(chat_history)>10:
            del chat_history[0:4]   
        txt=format_con(documents)
        ans=rag_chain.invoke({"question": cojoined, "context":txt})
        # h_grader=hallucination_grader.invoke({"generation":ans, "document":txt})
        # if h_grader=='yes':
        #     ans_grader=answer_grader.invoke({"generation":ans, "question":cojoined})
        #     # print("Reached hallucination")
        # else:
        #     # llm=ChatGroq(model="llama3-70b-8192", temperature=0)
        #     llm=ChatOpenAI(temperature=0)
        #     ans=rag_chain.invoke({"question": cojoined, "context":txt})
        #     ans_grader=answer_grader.invoke({"generation":ans, "question":cojoined})
            
        src=[]
        url=re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+',ans)
        if url != []:
            for urls in url:
                if urls.startswith("https://")== True:
                    src.append(urls)
            src="\n\n".join(src)
        else:
            src=" "
        if agent_ans=='Nil':
            pass
        else:
            ans=answer.invoke({"question":cojoined, "generation_1":agent_ans,
                   "generation_2":ans})
        ans_grader=answer_grader.invoke({"generation":ans, "question":cojoined})
        if ans_grader=='yes':
            # history.append((init_question,ans))
            with open("document.txt", "a", encoding="utf-8") as file:
                print(f"Question\n{init_question} \n\nAnswer\n{ans}\n\n Sources\n{src}\n\n", file=file)
            # return "", history,src
            return ans 
        else:
            # print("Not answered")
            # history.append((init_question,ans))
            with open("document.txt", "a", encoding="utf-8") as file:
                print(f"Question Not answered properly but below are the details of the question and the answer\n\nQuestion\n{init_question} \n\nAnswer\n{ans}\n\n Sources\n{src}\n\n", file=file)
            # return "", history, src
            return ans
        





# def get_response(msg):
    
#     return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)