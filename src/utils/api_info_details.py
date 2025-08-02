def get_api_info_details():
    return {
        "api_name": "Federal Student Loan Assistant",
        "version": "1.0.0",
        "description": "RAG-powered API for federal student loan customer service",
        "capabilities": {
            "knowledge_base": {
                "pdf_documents": [
                    "Academic Calendars, Cost of Attendance, and Packaging",
                    "Applications and Verification Guide",
                    "The Federal Pell Grant Program",
                    "The Direct Loan Program",
                ],
                "customer_data": "4,547 real customer complaints and scenarios",
                "total_documents": "Hybrid dataset with policy + complaint knowledge",
            },
            "retrieval_method": "Parent Document (best performing from RAGAS evaluation)",
            "evaluation_metrics": {
                "context_recall": "0.89",
                "faithfulness": "0.82",
                "answer_relevancy": "0.62",
                "factual_correctness": "0.41",
            },
        },
        "usage": {
            "endpoint": "/ask",
            "method": "POST",
            "example_questions": [
                "What are the requirements for federal student loan forgiveness?",
                "How do I apply for income-driven repayment plans?",
                "What should I do if my loan servicer is not responding?",
                "What are the differences between federal and private student loans?",
                "How does loan consolidation work?",
            ],
        },
        "response_format": {
            "answer": "Generated response text",
            "sources_count": "Number of knowledge sources used",
            "success": "Boolean indicating success",
            "message": "Status message",
        },
    }