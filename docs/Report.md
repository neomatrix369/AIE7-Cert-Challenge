# Task 1: Defining the Problem and Audience

**✅ Deliverables**

1. Write a succinct 1-sentence description of the problem

* **Problem:** Customer service representatives at federal student loan organizations such as https://www.mohela.com/ and https://studentaid.gov/ face significant challenges in delivering accurate and comprehensive responses to inquiries about student loan policies and repayment options from both current and prospective borrowers. On top of that is the rising need for catering to the niche needs of each such customer.

2. Write 1-2 paragraphs on why this is a problem for your specific user

* **Why:**  
  a. The resumption of student loan repayment during the summer months is anticipated to drive a substantial surge in customer service inquiries throughout 2025 [[1]](https://www.ed.gov/about/news/press-release/us-department-of-education-begin-federal-student-loan-collections-other-actions-help-borrowers-get-back-repayment) [[2]](https://www.cnbc.com/2025/07/15/interest-will-start-accruing-aug-1-for-student-loan-borrowers-on-save.html) - with interest resuming on SAVE plan loans in August 2025 and involuntary collections restarting after a 5-year pause
  b. The scale of this challenge is immense: as of the first quarter of 2025, outstanding student loan obligations total $1.777 trillion USD, with federal loans comprising $1.693 trillion of this burden and impacting 42.7 million Americans [[3]](https://educationdata.org/student-loan-debt-statistics) - representing approximately 12.5% of the U.S. population
  c. Customer service systems are already strained: the CFPB received over 18,000 student borrower complaints in 2024—the highest volume since 2012 [[4]](https://www.consumerfinance.gov/about-us/newsroom/cfpb-report-details-student-borrower-harms-from-servicing-failures-and-program-disruptions/) - with servicers facing extended call hold times, processing backlogs of 1.5 million IDR applications, and operational challenges during peak inquiry periods
  d. This issue carries personal significance for Dr. Greg, who is himself a student loan borrower, making this both a professionally relevant and personally meaningful problem to address.
  e. Everyone who calls up customer service goes through the same pains and routine process and if they strike luck might even get to the answer they were looking for. Providing customised or niche customer service on the subject of Student Loan at the federal agency isn't on the priority list yet
* **Success:** Reduced response time and improved accuracy for customer service representatives handling federal student loan inquiries
  1. 🧪 Hypothesis: Current customer service systems lack intelligent retrieval capabilities that can instantly access relevant policy documents and real customer scenarios, forcing agents to manually search through multiple systems and potentially provide incomplete or delayed responses
  2. 💡 Transformative solution: Deploy an intelligent agentic RAG system that dynamically adapts to specialized customer roles and niche scenarios, enabling scalable, role-based customer service that delivers precise, contextually-aware responses tailored to each borrower's unique situation
* **Audience:** Customer service representatives at federal student loan organizations (MOHELA, Nelnet, Aidvantage, Great Lakes) who handle diverse borrower inquiries daily. Our role-driven (persona-driven) approach provides specialized workflows for different borrower types that agents encounter: **Current Students** (enrollment and loan guidance), **Recent Graduates** (repayment transition support), **Active Borrowers** (payment management and modifications), **Public Service Workers** (PSLF eligibility and certification), **Financial Difficulty Cases** (hardship and default resolution), **Parent/Family** (PLUS loans and family guidance), **Disabled Students** (discharge options and accommodations), and **General Users** (basic information and servicer support). Each role provides curated example questions and contextually relevant tools through an intuitive chat interface.

---

# Task 2: Proposed Solution

**✅ Deliverables**

1. Write 1-2 paragraphs on your proposed solution.  How will it look and feel to the user?

* **Solution**: We are building an intelligent agentic RAG system that empowers customer service representatives to handle complex federal student loan inquiries with unprecedented speed and accuracy. The solution combines a hybrid knowledge base (federal policies + real customer complaints) with role-driven workflows to deliver contextually appropriate responses.

**How We Solve the Core Problem:**

| Problem Component | Our Solution Approach | Technical Implementation |
| :---- | ----- | ----- |
| **Generic responses** | 8 specialized borrower roles with curated question sets | Frontend role selection + context injection |
| **Manual system searches** | Intelligent retrieval with Parent-Document RAG (best RAGAS performer) | Hybrid knowledge base: 4 PDFs + 4,547 complaint records |
| **Incomplete information** | Multi-source orchestration with external tool integration | StudentAid.gov, MOHELA, and Tavily search APIs |
| **Delayed responses** | Sub-3-second response times with performance metrics | GPT-4 family + optimized vector search with Qdrant |

**Representative Scenarios We Handle:**

| Borrower Role | Complex Scenario | Solution Capability |
| :---- | ----- | ----- |
| **Active Borrower** | Customer is complaining that they are unable to pay and they are already 3 months behind on payments – what is the best solution? | Retrieves hardship options from policy docs + similar complaint resolutions with empathetic guidance |
| **Active Borrower** | Customer asserts that they did fill out an Income-Driven Repayment (IDR) plan renewal form, but there is none on record. Customer is insistent. What is the best solution? | Accesses servicer procedures + complaint patterns to provide step-by-step resolution paths |
| **Financial Difficulty** | Customer asserts "I shouldn't have to pay these back – this is unconstitutional." How do I respond? | Combines legal policy framework with de-escalation techniques from real complaint handling |
| **Current Student** | Is applying for and securing a student loan in 2025 a terrible idea? | Provides current market analysis, interest rates, and alternative funding sources |
| **Parent/Family** | How much loan money can I actually get from the government to go to school these days? Is there a cap? | Retrieves current borrowing limits, PLUS loan details, and dependency status impacts |
| **General User** | What grants and scholarships are available for free? | Searches federal aid database + external scholarship resources with eligibility criteria |

2. Describe the tools you plan to use in each part of your stack.  Write one sentence on why you made each tooling choice.
    1. LLM
    2. Embedding Model
    3. Orchestration
    4. Vector Database
    5. Monitoring
    6. Evaluation
    7. User Interface
    8. (Optional) Serving & Inference

Technology Stack Choices

- LLM: GPT-4 (gpt-4.1, gpt-4o-mini, gpt-4o-nano) - Excellent at understanding complex policy language and generating, and each of them have their expertise as per their individual strenghs. And all of them can perform customer-appropriate explanations while maintaining accuracy with citations. As Chris said they need to do a bit of thinky-thinky, hence ones that can reason well are include. For e.g. their strengths go from strong to week for these models respectively: pt-4.1, gpt-4o-mini, gpt-4o-nano
- Embedding Model: OpenAI `text-embedding-3-small` and `text-embedding-3-large` - the usual choice of a embeddings model with strong performance on documents and text
- Orchestration: LangGraph - Enables complex multi-agent workflows for our agentic RAG to run our tools, eventhough we have used only simple graphs
- Vector Database: Qdrant - as we have been using during our sessions and satisfies the basic requirements of storing chunks and searching documents, also code to use it integrated with the retrievers were already avail to use
- Monitoring: LangSmith - Built-in tracing for agent workflows, essential for debugging tool calls and measuring response quality in customer service context, in case we learn monitoring using LangSmith
- User Interface: Streamlit or Cursor/Claude Code generate front-end + Docker - rapid prototyping for agent-facing dashboard with real-time chat
- Evaluation: RAGAS - Industry standard for RAG evaluation with metrics that align with accuracy and relevance requirements for Student Loan responses

3. Where will you use an agent or agents?  What will you use “agentic reasoning” for in your app?

- _Simplified Agentic Reasoning Strategy: Smart Customer Service Assistant_
Primary Agent: Federal Student Loan Assistant
- Core Function: Intelligent question processing with context-aware tool selection and response generation
- Tool Arsenal: Policy and Complaints document retrieval, external search engine service(s) (loan specific sites and also general search engine searches)

Agentic Reasoning Within Single Agent: Smart Tool Orchestration & Contextual Routing

  The agent intelligently analyzes each user query to determine its type and complexity. It then orchestrates a mix of
  internal and external tools:

   * Internal RAG Tools: Prioritized for questions answerable by the project's hybrid knowledge base (federal policies,
      customer complaints). This includes specialized RAG method like Parent-Document (only using one method as per RAGAS Evaluation outcome)
   * External Search Tools: Utilized for information beyond the internal dataset, such as official federal guidance
     (StudentAid.gov), servicer-specific details (Mohela), comprehensive loan comparison service (StudentAid & Mohela combined), or broader web searches (using Tavily).

  This dynamic routing ensures the most relevant and authoritative information source is consulted, adapting to the query's specific needs.

Why Single Agent Works?:
- Sufficient Complexity: The agentic behavior comes from intelligent tool orchestration and orchestrating multiple tools, not from multiple specialized agents
- Clear Workflow: _Question_ → _Analyze_ → _Retrieve_ → _Synthesize_ → _Respond_ (with escalation branching) - linear enough for one agent to handle effectively

---

# Task 3: Dealing with the Data

**✅ Deliverables**

1. Describe all of your data sources and external APIs, and describe what you’ll use them for.

_Data Sources_

All the files in the `data` folder are our dataset, shared during the covert:
- four PDF files 
  - Academic_Calenders_Cost_of_Attendance_and_Packaging.pdf
  - Applications_and_Verification_Guide.pdf
  - The_Direct_Loan_Program.pdf
  - The_Federal_Pell_Grant_Program.pdf
- one complains.csv file (4.5K records)

2. Describe the default chunking strategy that you will use.  Why did you make this decision?

_Chunking strategy_

The simplest chunking strategy has been used:
```python
### taking advise from and also reviewing all the code and notebooks shared on RAG and works (legacy implementation)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
```
One of the analysis from an LLM suggested for the type of data/documents present in the `data` folder the below is the most optimised one to have for the child splitter:
```python
child_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
```

Although the above config didn't help but deteriorated the metrics (scores) for the Parent Docoment retriever.

While the Parent Docoment retriever can be setup this way:
```python
parent_text_splitter = RecursiveCharacterTextSplitter(chunk_size=750 chunk_overlap=100) ### as other retrievers and same as child splitter

child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=750 chunk_overlap=100)
```

---

# Task 4: Building a Quick End-to-End Agentic RAG Prototype

**✅ Deliverables**
- Build an end-to-end prototype and deploy it to a local endpoint
See [AIE7-Cert-Challenge](https://github.com/neomatrix369/AIE7-Cert-Challenge) | [README](https://github.com/neomatrix369/AIE7-Cert-Challenge/blob/main/README.md) | [Frontend README](https://github.com/neomatrix369/AIE7-Cert-Challenge/blob/main/frontend/README.md) | [Backend README](https://github.com/neomatrix369/AIE7-Cert-Challenge/blob/main/src/backend/README.md)

---
![front-page-screenshot](../screenshots/front-page-screenshot.png)
---
**LangSmith monitoring:** [LangSmith Main page](../screenshots/LangSmith%20monitoring%20Main%20Page.jpg) | [LangSmith: ask_parent_document_llm_tool tool)](../screenshots/LangSmith%20monitoring%20ask_parent_document_llm_tool%20tool.jpg) | [LangSmith: StudentAid_Federal_Search tool](../screenshots/LangSmith%20monitoring%20StudentAid_Federal_Search%20tool.jpg)

**Other screenshots:** [Swagger UI](../screenshots/swagger-ui-screenshot.png) | [Frontend blocking terminal/console](../screenshots/terminal-screen-frontend-app.jpg) | [Backend blocking terminal/console](../screenshots/terminal-screen-backend-app.jpg) 

# Task 5: Creating a Golden Test Data Set

**✅ Deliverables**
1. Assess your pipeline using the RAGAS framework including key metrics faithfulness, response relevance, context precision, and context recall.  Provide a table of your output results.

2. What conclusions can you draw about the performance and effectiveness of your pipeline with this information?

# Task 6: The Benefits of Advanced Retrieval

**✅ Deliverables**
1. Describe the retrieval techniques that you plan to try and to assess in your application. Write one sentence on why you believe each technique will be useful for your use case.

2. Test a host of advanced retrieval techniques on your application.

# Task 7: Assessing Performance

**✅ Deliverables**
1. How does the performance compare to your original RAG application?  Test the advanced retrieval method using the RAGAS frameworks to quantify any improvements.  Provide results in a table.


2. Articulate the changes that you expect to make to your app in the second half of the course. How will you improve your application?

A few things I would change to improve my application:
- functionally I would add a way to save sessions, develop a question-autocomplete features, show a list of most popular questions (asked by others per role)
- device a way to measure run-time performance of the RAG application
- apply Guard-rails to input (entry points) and output (exit points) -- using the guardrails.ai package
- improve the evaluation framework to be able to perform runs/experiments/tests controlled by changing parameters (a bit like hyper-parameter tuning)