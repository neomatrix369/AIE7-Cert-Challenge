## Task 1: Defining the Problem and Audience

* **Problem:** It is hard to get clear, precise answers to questions about student loans and student loan repayment for existing borrowers and potential borrowers, even as a customer service agent within a federal student loan company like [Mohela](https://www.mohela.com/) and [Federal Student Aid](https://studentaid.gov/).

* **Why:**  
  1. As student loan repayment continues this summer, the volume of calls is expected to increase in 2025.  
  2. As of Q1, 2025, there is $1.777T USD in student loan debt, including $1.693T in Federal student loan debt, affecting 42.7M people in the US.  
  3. I (Dr. Greg) also have student loans, and so this is a problem that is interesting and important to me.  
* **Success:** Time saved by responding to customer inquiries and complaints.  
  1. 🧪 Hypothesis: The existing tooling used for search, retrieval, and generation of helpful and useful answers to customer inquiries does not rapidly accelerate an agent’s ability to deal with a large volume of customer complaints per day.  
* **Audience:** We are building this application for customer service agents who work in Federal Student Loan companies.

## Task 2: Proposed Solution
* **Solution**: We are building a solution that can answer questions like:

| Category | Scenario / Question |
| :---- | ----- |
| **Existing borrowers in repayment** | Customer is complaining that they are unable to pay and they are already 3 months behind on payments – what is the best solution? |
| **Existing borrowers in repayment** | Customer asserts that they did fill out an Income-Driven Repayment (IDR) plan renewal form, but there is none on record. Customer is insistent. What is the best solution? |
| **Existing borrowers in repayment** | Customer asserts “I shouldn’t have to pay these back – this is unconstitutional.” How do I respond? |
| **New borrowers** | Is applying for and securing a student loan in 2025 a terrible idea? |
| **New borrowers** | How much loan money can I actually get from the government to go to school these days? Is there a cap? |
| **New borrowers** | What grants and scholarships are available for free? |


Technology Stack Choices

- LLM: GPT-4 (any one from the family) - Excellent at understanding complex policy language and generating customer-appropriate explanations while maintaining accuracy with citations.
- Embedding Model: OpenAI text-embedding-3-large - Strong performance on domain-specific content with good retrieval precision for policy documents and regulatory text.
- Orchestration: LangGraph - Enables complex multi-agent workflows for borrower classification, policy retrieval, and response validation with clear decision paths.
- Vector Database: Qdrant - as we have been using during our sessions and satisfies the basic requirements of storing and searching documents
- Monitoring: LangSmith - Built-in tracing for agent workflows, essential for debugging tool calls and measuring response quality in customer service context.
- User Interface: Streamlit or Cursor generate front-end - Rapid prototyping for agent-facing dashboard with real-time chat
- Evaluation: RAGAS - Industry standard for RAG evaluation with metrics that align with accuracy and relevance requirements for policy-based responses.
- Serving & Inference: FastAPI + Docker - Production-ready deployment with session management between user sessions

- _Simplified Agentic Reasoning Strategy: Smart Customer Service Assistant_
Primary Agent: Federal Student Loan Assistant
- Core Function: Intelligent question processing with context-aware tool selection and response generation
- Tool Arsenal: Policy document retrieval, external search engine service

Agentic Reasoning Within Single Agent:
- Question Classification & Tool Selection: Determines question type (repayment, eligibility, procedures, others) and selects appropriate tools 
- Routes to policy documents vs. external search based on question complexity

Why Single Agent Works:
- Sufficient Complexity: The agentic behavior comes from intelligent tool orchestration and multi-step reasoning within one agent, not from multiple specialized agents.
- Clear Workflow: Question → Analyze → Retrieve → Synthesize → Respond (with escalation branching) - linear enough for one agent to handle effectively.

## Task 3: Dealing with the Data

Is the complains dataset at https://github.com/neomatrix369/AIE7/blob/s09-assignment/09_Advanced_Retrieval/data/complaints.csv sufficient?

**✅ Deliverables**
1. Describe all of your data sources and external APIs, and describe what you’ll use them for.
2. Describe the default chunking strategy that you will use.  Why did you make this decision?
3. [Optional] Will you need specific data for any other part of your application?   If so, explain.

## Task 4: Building a Quick End-to-End Agentic RAG Prototype

**✅ Deliverables**
- Build an end-to-end prototype and deploy it to a local endpoint

Task 5: Creating a Golden Test Data Set

**✅ Deliverables**
1. Assess your pipeline using the RAGAS framework including key metrics faithfulness, response relevance, context precision, and context recall.  Provide a table of your output results.
2. What conclusions can you draw about the performance and effectiveness of your pipeline with this information?

Task 6: The Benefits of Advanced Retrieval

**✅ Deliverables**
1. Describe the retrieval techniques that you plan to try and to assess in your application. Write one sentence on why you believe each technique will be useful for your use case.
2. Test a host of advanced retrieval techniques on your application.

Task 7: Assessing Performance

**✅ Deliverables**
1. How does the performance compare to your original RAG application?  Test the fine-tuned embedding model using the RAGAS frameworks to quantify any improvements.  Provide results in a table.
2. Articulate the changes that you expect to make to your app in the second half of the course. How will you improve your application?