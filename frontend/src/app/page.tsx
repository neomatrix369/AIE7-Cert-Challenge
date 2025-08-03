'use client'

import { useState, useEffect, useRef } from 'react'
import { Send, Clock, CheckCircle, AlertCircle, Info, GraduationCap, ChevronDown, ChevronUp } from 'lucide-react'

interface PerformanceMetrics {
  response_time_ms?: number;
  retrieval_time_ms?: number;
  generation_time_ms?: number;
  confidence_score?: number;
  tokens_used?: number;
  input_tokens?: number;
  output_tokens?: number;
}

interface Message {
  id: string
  content: string
  isUser: boolean
  timestamp: Date
  sources?: number
  processingTime?: string
  performance_metrics?: PerformanceMetrics
  source_details?: any[]
  focus?: string; // Added focus to Message interface
}

interface Question {
  text: string;
  focus: string;
}

interface Persona {
  id: string
  name: string
  emoji: string
  description: string
  color: string
  questions: Question[]
}

const PERSONAS: Persona[] = [
  {
    id: 'current-student',
    name: 'Current Student',
    emoji: '🎓',
    description: 'Currently enrolled in school, exploring loan options',
    color: 'bg-blue-100 border-blue-300 text-blue-800',
    questions: [
      { text: 'How much can I borrow for my degree program?', focus: 'General' },
      { text: 'What\'s the difference between subsidized and unsubsidized loans?', focus: 'General' },
      { text: 'When do I need to start repaying my loans?', focus: 'General' },
      { text: 'Can I use loans for living expenses?', focus: 'General' },
      { text: 'What is the Student Aid Index (SAI), and how does it affect my eligibility for federal student aid?', focus: 'FAFSA' },
      { text: 'Can you explain how my academic calendar and payment periods influence when I receive my federal student loan disbursements?', focus: 'Loan Disbursements' },
      { text: 'What are the key differences between Direct Subsidized and Unsubsidized Loans, and which one is more suitable for my situation?', focus: 'Loan Types' },
    ]
  },
  {
    id: 'recent-graduate',
    name: 'Recent Graduate',
    emoji: '🎯',
    description: 'Recently graduated, entering repayment phase',
    color: 'bg-green-100 border-green-300 text-green-800',
    questions: [
      { text: 'What are my repayment options after graduation?', focus: 'General' },
      { text: 'How do income-driven repayment plans work?', focus: 'General' },
      { text: 'Can I consolidate my federal student loans?', focus: 'General' },
      { text: 'What happens during my grace period?', focus: 'General' },
      { text: 'What are my repayment options for federal student loans after my grace period ends, and how do I choose the best one?', focus: 'Repayment Options' },
      { text: 'I\'m considering consolidating my federal student loans. What are the pros and cons, and how does the process work?', focus: 'Loan Consolidation' },
      { text: 'My loan servicer is reporting missed payments from my grace period, but I thought payments weren\'t due yet. What should I do?', focus: 'Servicer Issues' },
    ]
  },
  {
    id: 'parent-family',
    name: 'Parent/Family',
    emoji: '👨‍👩‍👧‍👦',
    description: 'Parent or family member helping with education financing',
    color: 'bg-purple-100 border-purple-300 text-purple-800',
    questions: [
      { text: 'Should I take a PLUS loan or help my child borrow?', focus: 'General' },
      { text: 'What are the credit requirements for Parent PLUS loans?', focus: 'General' },
      { text: 'How can I help my child understand their loan responsibilities?', focus: 'General' },
      { text: 'What tax benefits are available for education expenses?', focus: 'General' },
      { text: 'As a parent, what information do I need to provide on the FAFSA, and how does my income affect my child\'s financial aid?', focus: 'FAFSA' },
      { text: 'What are the eligibility requirements and responsibilities for taking out a Direct PLUS Loan for my child\'s education?', focus: 'PLUS Loans' },
      { text: 'I\'m concerned about the privacy of my personal and financial data related to my child\'s student loans. What protections are in place, and what if there\'s a data breach?', focus: 'Data Privacy' },
    ]
  },
  {
    id: 'active-borrower',
    name: 'Active Borrower',
    emoji: '💰',
    description: 'Currently repaying loans, managing payments',
    color: 'bg-orange-100 border-orange-300 text-orange-800',
    questions: [
      { text: 'Can I lower my monthly payment amount?', focus: 'General' },
      { text: 'How do I apply for loan forgiveness programs?', focus: 'General' },
      { text: 'What should I do if I\'m having trouble making payments?', focus: 'General' },
      { text: 'Can I pay off my loans early without penalties?', focus: 'General' },
      { text: 'My monthly student loan payments have increased unexpectedly. How can I understand the reason for this change and explore options to lower them?', focus: 'Payment Management' },
      { text: 'I\'ve made payments, but my loan balance doesn\'t seem to be decreasing, or payments aren\'t being applied correctly. How can I get a full accounting of my loan and resolve these discrepancies?', focus: 'Loan Accounting' },
      { text: 'My credit score was negatively impacted due to a reported delinquency, but I believe it was an error by my servicer. What steps can I take to dispute this and have it corrected?', focus: 'Credit Reporting' },
    ]
  },
  {
    id: 'public-service',
    name: 'Public Service Worker',
    emoji: '🏛️',
    description: 'Working in public service, eligible for PSLF',
    color: 'bg-indigo-100 border-indigo-300 text-indigo-800',
    questions: [
      { text: 'Do I qualify for Public Service Loan Forgiveness?', focus: 'General' },
      { text: 'How do I certify my employment for PSLF?', focus: 'General' },
      { text: 'What payment plan is best for PSLF?', focus: 'General' },
      { text: 'How many qualifying payments have I made?', focus: 'General' },
      { text: 'What are the specific criteria for Public Service Loan Forgiveness (PSLF), and how do I ensure my employment qualifies?', focus: 'PSLF Eligibility' },
      { text: 'How do I certify my employment for PSLF, and what should I do if my servicer isn\'t correctly tracking my qualifying payments?', focus: 'PSLF Certification' },
      { text: 'Can you explain the process for consolidating loans for PSLF, and how does it affects my payment count?', focus: 'PSLF Consolidation' },
    ]
  },
  {
    id: 'financial-difficulty',
    name: 'Financial Difficulty',
    emoji: '⚠️',
    description: 'Experiencing financial hardship with loan payments',
    color: 'bg-red-100 border-red-300 text-red-800',
    questions: [
      { text: "I can't make my loan payments, what are my options?", focus: 'General' },
      { text: 'How do I apply for deferment or forbearance?', focus: 'General' },
      { text: 'What happens if I default on my federal student loans?', focus: 'General' },
      { text: 'How can I rehabilitate defaulted loans?', focus: 'General' },
      { text: "I'm struggling to make my student loan payments. What are my options for temporary relief, such as forbearance or deferment?", focus: 'Financial Hardship' },
      { text: 'What are the consequences of defaulting on a federal student loan, and what steps can I take to rehabilitate or resolve a defaulted loan?', focus: 'Loan Default' },
      { text: 'I believe I qualify for a loan discharge due to school closure or misrepresentation. What is the process for applying for borrower defense to repayment?', focus: 'Loan Discharge' },
    ]
  }
]

interface ApiResponse {
  answer: string
  sources_count: number
  success: boolean
  message?: string
  performance_metrics?: PerformanceMetrics
  source_details?: any[]
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState<'unknown' | 'healthy' | 'unhealthy'>('unknown')
  const [selectedPersona, setSelectedPersona] = useState<Persona | null>(null)
  const [showPersonaSelection, setShowPersonaSelection] = useState(true)
  const [showExampleQuestionsSlider, setShowExampleQuestionsSlider] = useState(true) // New state for slider
  const [selectedQuestionFocus, setSelectedQuestionFocus] = useState<string | null>(null); // New state for question focus
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages]);

  useEffect(() => {
    checkApiHealth()
  }, [])

  const selectPersona = (persona: Persona) => {
    setSelectedPersona(persona)
    setShowPersonaSelection(false)
    setMessages([])
    setShowExampleQuestionsSlider(true) // Show slider when persona is selected
  }

  const resetPersona = () => {
    setSelectedPersona(null)
    setShowPersonaSelection(true)
    setMessages([])
    setInputMessage('')
    setShowExampleQuestionsSlider(true) // Show slider when persona is reset
  }

  const checkApiHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/health')
      const data = await response.json()
      setApiStatus(data.status === 'healthy' ? 'healthy' : 'unhealthy')
      console.log('API Health Check: ', data.status)
    } catch (error) {
      setApiStatus('unhealthy')
      console.error('API Health Check Failed: ', error);
      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        console.error('Details: Network error or backend is not running.');
      } else {
        console.error(`Details: ${String(error)}`);
      }
    }
  }

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    console.log('Frontend: User message sent', inputMessage)

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      isUser: true,
      timestamp: new Date(),
      focus: selectedQuestionFocus || undefined // Include focus in user message
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      console.log('Frontend: Sending request to backend /ask endpoint')
      const requestBody: any = {
        question: inputMessage,
        max_response_length: 2000,
      };

      if (selectedPersona) {
        requestBody.persona = {
          id: selectedPersona.id,
          name: selectedPersona.name,
          description: selectedPersona.description
        };
      }

      if (selectedQuestionFocus) {
        requestBody.focus = selectedQuestionFocus;
      }

      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) {
        let errorDetail = 'Unknown error';
        try {
          const errorBody = await response.json();
          errorDetail = errorBody.detail || JSON.stringify(errorBody);
        } catch (parseError) {
          errorDetail = response.statusText || 'Could not parse error response';
        }
        console.error(
          `Frontend: Backend response not OK. Status: ${response.status}, Details: ${errorDetail}`
        );
        throw new Error(`HTTP error! Status: ${response.status}, Details: ${errorDetail}`);
      }

      const data: ApiResponse = await response.json()
      console.log('Frontend: Backend Response Received:', data)

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.answer,
        isUser: false,
        timestamp: new Date(),
        sources: data.sources_count,
        processingTime: data.performance_metrics?.response_time_ms
          ? `${(data.performance_metrics.response_time_ms / 1000).toFixed(2)}s` 
          : undefined,
        performance_metrics: data.performance_metrics,
        source_details: data.source_details
      }

      setMessages(prev => [...prev, botMessage])
      console.log('Frontend: Bot message displayed')
    } catch (error) {
      console.error('Frontend: Error during message sending:', error);
      let errorMessageText = 'Sorry, I encountered an error while processing your question.';
      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        errorMessageText += '\n\nThis usually means the backend server is not running or is inaccessible.\nPlease ensure the backend API is running on http://localhost:8000.';
      } else if (error instanceof Error) {
        errorMessageText += `\n\nDetails: ${error.message}`;
      } else {
        errorMessageText += `\n\nDetails: ${String(error)}`;
      }

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: errorMessageText,
        isUser: false,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
      console.log('Frontend: Loading state set to false')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleExampleQuestion = (question: Question) => {
    // Populate input field with focus and question for user review
    const formattedInput = `Focus: ${question.focus}\n${question.text}`;
    setInputMessage(formattedInput);
    setSelectedQuestionFocus(question.focus);
    setShowExampleQuestionsSlider(false); // Collapse slider on question selection
  };

  return (
    <div className="min-h-screen bg-[var(--background)] flex flex-col">
      <header className="bg-blue-600 text-white p-4 shadow-md w-full">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <GraduationCap className="h-8 w-8" />
            <h1 className="text-xl font-semibold">Federal Student Loan Assistant</h1>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-sm ${apiStatus === 'healthy' ? 'bg-green-500' : 'bg-red-500'}`}>
              {apiStatus === 'healthy' ? <CheckCircle className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
              <span>{apiStatus === 'healthy' ? 'Online' : 'Offline'}</span>
            </div>
          </div>
        </div>
      </header>

      <div className="flex-1 max-w-6xl mx-auto w-full p-4">
        {showPersonaSelection ? (
          <div className="py-4">
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-6xl mx-auto">
              {PERSONAS.map((persona) => (
                <button
                  key={persona.id}
                  onClick={() => selectPersona(persona)}
                  className={`p-4 rounded-lg border-2 transition-all duration-200 hover:shadow-lg hover:scale-105 text-left ${persona.color}`}
                >
                  <div className="text-3xl mb-2">{persona.emoji}</div>
                  <h3 className="font-semibold text-lg mb-1">{persona.name}</h3>
                  <p className="text-sm opacity-80">{persona.description}</p>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto">
            {selectedPersona && (
              <div className="mb-4 p-4 rounded-lg bg-blue-50 border border-blue-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{selectedPersona.emoji}</span>
                    <div>
                      <span className="font-medium text-blue-900">Welcome, {selectedPersona.name}!</span>
                      <p className="text-sm text-blue-700">I'm here to help with federal student loan questions tailored to your situation.</p>
                    </div>
                  </div>
                  <button
                    onClick={resetPersona}
                    className="text-sm text-blue-600 hover:text-blue-800 underline whitespace-nowrap"
                  >
                    Change Role
                  </button>
                </div>
              </div>
            )}
            
            <div className="space-y-4 mb-4">
              {messages.map((message) => (
                <div
              key={message.id}
              className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`relative max-w-3xl rounded-lg p-4 shadow-md ${message.isUser ? 'bg-[var(--user-message-bg)] text-[var(--foreground)]' : 'bg-[var(--bot-message-bg)] text-[var(--foreground)] border border-[var(--border)]'}`}>
                <div className="whitespace-pre-wrap">{message.content}</div>
                <div className="flex items-center justify-between mt-2 text-xs opacity-70">
                  <span>{message.timestamp.toLocaleTimeString()}</span>
                  {!message.isUser && (
                    <div className="flex items-center space-x-3">
                      {message.sources && (
                        <span className="flex items-center space-x-1 group/sources relative">
                          <span className="cursor-help">📚 {message.sources} sources</span>
                          {message.source_details && message.source_details.length > 0 && (
                            <div className="absolute right-0 bottom-full mb-2 w-64 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-2 text-xs text-gray-700 dark:text-gray-300 opacity-0 group-hover/sources:opacity-100 transition-opacity duration-300 z-10">
                              <h4 className="font-semibold mb-1">Sources:</h4>
                              <ul className="list-disc list-inside max-h-40 overflow-y-auto">
                                {message.source_details.map((source, index) => {
                                  const content = typeof source === 'object' && source.content ? source.content : (typeof source === 'string' ? source : JSON.stringify(source));
                                  const truncatedContent = content.length > 80 ? content.substring(0, 80) + '...' : content;
                                  return (
                                    <li key={index} className="mb-1 text-xs">
                                      {typeof source === 'object' && source.relevance_score !== undefined && source.relevance_score > 0 ? (
                                        <span>
                                          <span className="font-medium text-blue-600">
                                            [{source.relevance_score.toFixed(3)}]
                                          </span>
                                          {' '}
                                          <span>{truncatedContent}</span>
                                        </span>
                                      ) : (
                                        <span>{truncatedContent}</span>
                                      )}
                                    </li>
                                  );
                                })}
                              </ul>
                            </div>
                          )}
                        </span>
                      )}
                      {message.processingTime && (
                        <span className="flex items-center space-x-1">
                          <Clock className="h-3 w-3" />
                          <span>{message.processingTime}</span>
                        </span>
                      )}
                    </div>
                  )}
                </div>
                {!message.isUser && message.performance_metrics && (
                  <div className="group absolute top-2 right-2">
                    <div className="bg-[var(--primary)] hover:bg-[var(--primary-hover)] rounded-full p-1 transition-colors duration-200 cursor-pointer">
                      <Info className="h-3 w-3 text-white" />
                    </div>
                    <div className="absolute right-0 bottom-full mb-2 w-64 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-2 text-xs text-gray-700 dark:text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-10">
                      <h3 className="font-semibold mb-1">Performance & Source Details</h3>
                      {message.performance_metrics.response_time_ms && <p>Response Time: {(message.performance_metrics.response_time_ms / 1000).toFixed(2)}s</p>}
                      {message.performance_metrics.retrieval_time_ms && <p>Retrieval Time: {(message.performance_metrics.retrieval_time_ms / 1000).toFixed(2)}s</p>}
                      {message.performance_metrics.generation_time_ms && <p>Generation Time: {(message.performance_metrics.generation_time_ms / 1000).toFixed(2)}s</p>}
                      {message.performance_metrics.tokens_used && <p>Tokens Used: {message.performance_metrics.tokens_used}</p>}
                      {message.source_details && message.source_details.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-gray-300 dark:border-gray-600">
                          <h4 className="font-semibold">Sources:</h4>
                          <ul className="list-disc list-inside">
                            {message.source_details.map((source, index) => (
                              <li key={index} className="truncate">
                                {typeof source === 'object' && source.relevance_score !== undefined && source.relevance_score > 0 ? (
                                  <span>
                                    <span className="font-medium text-blue-600">
                                      [{source.relevance_score.toFixed(3)}]
                                    </span>
                                    {' '}
                                    {source.content}
                                  </span>
                                ) : (
                                  <span>
                                    {typeof source === 'object' && source.content ? source.content : (typeof source === 'string' ? source : JSON.stringify(source))}
                                  </span>
                                )}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {messages.length <= 1 && !isLoading && selectedPersona && (
          <div className="text-center text-gray-500 dark:text-gray-400">
            <h2 className="text-lg font-semibold mb-2">Here are some questions relevant to {selectedPersona.name.toLowerCase()}s:</h2>
            <div className="relative w-full">
              <div 
                className={`flex flex-col space-y-2 p-2 transition-all duration-300 ease-in-out ${showExampleQuestionsSlider ? 'max-h-[60vh] opacity-100 overflow-y-auto' : 'max-h-0 opacity-0 overflow-hidden'}`}
                style={{ 
                  scrollbarWidth: 'thin', 
                  scrollbarColor: '#cbd5e1 transparent'
                } as React.CSSProperties}
              >
                {selectedPersona.questions.map((question, index) => (
                  <button 
                    key={index}
                    onClick={() => handleExampleQuestion(question)} 
                    className="flex-none bg-[var(--bot-message-bg)] hover:bg-gray-100 p-3 rounded-lg text-sm text-[var(--foreground)] text-left border border-[var(--border)] shadow-sm"
                  >
                    <span className="font-semibold">Focus: {question.focus}</span><br/>{question.text}
                  </button>
                ))}
              </div>
              <button
                onClick={() => setShowExampleQuestionsSlider(!showExampleQuestionsSlider)}
                className="absolute top-0 right-0 p-1 bg-gray-200 dark:bg-gray-700 rounded-full shadow-md hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-600 dark:text-gray-300"
                title={showExampleQuestionsSlider ? 'Collapse suggestions' : 'Expand suggestions'}
              >
                {showExampleQuestionsSlider ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
              </button>
            </div>
          </div>
        )}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-[var(--bot-message-bg)] text-[var(--foreground)] shadow-md border border-[var(--border)] rounded-lg p-4 max-w-3xl">
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-[var(--primary)]"></div>
                <span>Searching through federal student loan knowledge base...</span>
              </div>
            </div>
          </div>
        )}
          </div>
        )}
      </div>

      <div className="border-t border-[var(--border)] bg-[var(--background)] p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex space-x-4">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask me anything about federal student loans..."
              className="flex-1 border border-gray-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent resize-none bg-white text-black"
              rows={2}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="px-6 py-3 bg-[var(--primary)] text-white rounded-lg hover:bg-[var(--primary-hover)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)] disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2">
              <Send className="h-4 w-4" />
              <span>Send</span>
            </button>
          </div>
          <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
            Press Enter to send, Shift+Enter for new line
          </div>
        </div>
      </div>
    </div>
  )
}