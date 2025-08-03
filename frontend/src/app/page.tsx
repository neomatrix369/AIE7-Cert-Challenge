'use client'

import { useState, useEffect, useRef } from 'react'
import { Send, Clock, CheckCircle, AlertCircle, Info, GraduationCap, ChevronDown, ChevronUp, X } from 'lucide-react'

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
  tools_used?: string[]
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
  },
  {
    id: 'general-user',
    name: 'General User',
    emoji: '👤',
    description: 'Seeking quick answers and general information',
    color: 'bg-gray-100 border-gray-300 text-gray-800',
    questions: [
      { text: 'What types of federal student loans are available?', focus: 'Basic Info' },
      { text: 'How do I apply for federal student aid?', focus: 'Application Process' },
      { text: 'What is the difference between federal and private student loans?', focus: 'Loan Comparison' },
      { text: 'Where can I find my loan servicer contact information?', focus: 'Servicer Info' },
      { text: 'What should I do if I\'m having trouble with my loan servicer?', focus: 'Servicer Issues' },
      { text: 'How can I check my federal student loan balance and payment history?', focus: 'Account Access' },
      { text: 'What resources are available for understanding my repayment options?', focus: 'Resources' },
    ]
  },
  {
    id: 'disabled-student',
    name: 'Disabled Student',
    emoji: '♿',
    description: 'Seeking accessible information and support options',
    color: 'bg-rose-100 border-rose-300 text-rose-800',
    questions: [
      { text: 'What loan discharge options are available for students with disabilities?', focus: 'Disability Discharge' },
      { text: 'How do I apply for Total and Permanent Disability discharge?', focus: 'TPD Discharge' },
      { text: 'What documentation is needed to prove disability for loan discharge?', focus: 'Documentation' },
      { text: 'Are there special repayment options for borrowers with disabilities?', focus: 'Repayment Options' },
      { text: 'What accommodations are available for borrowers who need accessible communication formats?', focus: 'Accessibility' },
      { text: 'How can I get help completing loan forms if I have a disability that affects my ability to process paperwork?', focus: 'Form Assistance' },
      { text: 'What happens to my loans if my disability status changes after receiving a discharge?', focus: 'Status Changes' },
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
  tools_used?: string[]
}

// Helper function to convert technical tool names to human-readable names
const getHumanReadableToolName = (toolName: string): string => {
  const toolNameMap: { [key: string]: string } = {
    // RAG Tools - Federal Student Loan Knowledge Base
    'ask_parent_document_llm_tool': '📄 Advanced Document Search',
    'ask_contextual_compression_llm_tool': '🎯 Precision Context Filter', 
    'ask_multi_query_llm_tool': '🔍 Comprehensive Query Expansion',
    'ask_naive_llm_tool': '📚 Standard Document Search',

    // External Search Tools
    'tavily_search': '🌐 General Web Search',
    'studentaid_search': '🏛️ Federal Student Aid Database',
    'mohela_search': '🏢 MOHELA Servicer Portal',

    // Alternative naming patterns that might be used
    'parent_document': '📄 Advanced Document Search',
    'contextual_compression': '🎯 Precision Context Filter',
    'multi_query': '🔍 Comprehensive Query Expansion',
    'naive': '📚 Standard Document Search',
    'basic_search': '📚 Standard Document Search',
    'advanced_search': '📄 Advanced Document Search',
    'web_search': '🌐 General Web Search'
  };

  // Handle variations and clean up the tool name
  const cleanedName = toolName.toLowerCase().replace(/[_-]/g, '_');
  return toolNameMap[cleanedName] || toolNameMap[toolName] || `🔧 ${toolName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}`;
};

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState<'unknown' | 'healthy' | 'unhealthy'>('unknown')
  const [selectedPersona, setSelectedPersona] = useState<Persona | null>(null)
  const [showPersonaSelection, setShowPersonaSelection] = useState(true)
  const [showExampleQuestionsSlider, setShowExampleQuestionsSlider] = useState(true) // New state for slider
  const [selectedQuestionFocus, setSelectedQuestionFocus] = useState<string | null>(null); // New state for question focus
  
  // Chat session persistence - store messages for each persona
  const [chatSessions, setChatSessions] = useState<{ [personaId: string]: Message[] }>({})
  
  // Tooltip visibility state
  const [openTooltips, setOpenTooltips] = useState<{ [messageId: string]: { 
    info?: boolean, 
    sources?: boolean,
    infoPosition?: 'above' | 'below',
    sourcesPosition?: 'above' | 'below'
  } }>({})
  
  
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const toggleTooltip = (messageId: string, type: 'info' | 'sources', event?: React.MouseEvent) => {
    if (event && !openTooltips[messageId]?.[type]) {
      // When opening tooltip, check if we need smart positioning
      const rect = event.currentTarget.getBoundingClientRect()
      const viewportHeight = window.innerHeight
      const spaceAbove = rect.top
      const spaceBelow = viewportHeight - rect.bottom
      
      // If not enough space above and more space below, show below
      const shouldShowBelow = spaceAbove < 200 && spaceBelow > spaceAbove
      
      setOpenTooltips(prev => ({
        ...prev,
        [messageId]: {
          ...prev[messageId],
          [type]: true,
          [`${type}Position`]: shouldShowBelow ? 'below' : 'above'
        }
      }))
    } else {
      // Toggle or close
      setOpenTooltips(prev => ({
        ...prev,
        [messageId]: {
          ...prev[messageId],
          [type]: !prev[messageId]?.[type]
        }
      }))
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages]);

  useEffect(() => {
    checkApiHealth()
  }, [])

  // Save messages to current persona's session whenever messages change
  useEffect(() => {
    if (selectedPersona && messages.length > 0) {
      setChatSessions(prev => ({
        ...prev,
        [selectedPersona.id]: messages
      }))
    }
  }, [messages, selectedPersona])

  const selectPersona = (persona: Persona) => {
    // Save current messages to the current persona's session (if any)
    if (selectedPersona) {
      setChatSessions(prev => ({
        ...prev,
        [selectedPersona.id]: messages
      }))
    }
    
    // Load saved messages for the new persona (or empty array)
    const savedMessages = chatSessions[persona.id] || []
    
    setSelectedPersona(persona)
    setShowPersonaSelection(false)
    setMessages(savedMessages)
    setShowExampleQuestionsSlider(true) // Show slider when persona is selected
  }

  const resetPersona = () => {
    // Save current messages to the current persona's session before resetting
    if (selectedPersona) {
      setChatSessions(prev => ({
        ...prev,
        [selectedPersona.id]: messages
      }))
    }
    
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
      console.log('Frontend: Performance Metrics:', data.performance_metrics)
      console.log('Frontend: Tools Used:', data.tools_used)

      // Debug: Test tool name mapping
      if (data.tools_used) {
        console.log('Frontend: Human-readable tool names:',
          data.tools_used.map(tool => `${tool} → ${getHumanReadableToolName(tool)}`)
        )
      }

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
        source_details: data.source_details,
        tools_used: data.tools_used
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

  const clearInput = () => {
    setInputMessage('')
    setSelectedQuestionFocus(null)
  }

  const handleExampleQuestion = (question: Question) => {
    // Populate input field with role, focus and question for user review
    const formattedInput = `Role: ${selectedPersona?.name}\nFocus: ${question.focus}\n${question.text}`;
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

      <div className="flex-1 max-w-6xl mx-auto w-full p-4 overflow-y-auto">
        {showPersonaSelection ? (
          <div className="py-8">
            {/* Welcome Section */}
            <div className="text-center mb-8 max-w-4xl mx-auto">
              <h2 className="text-3xl font-bold text-gray-900 mb-3">
                🎯 What's Your Student Loan Situation?
              </h2>
              <p className="text-lg text-gray-600 mb-6">
                Choose your role below to get personalized assistance with your federal student loans
              </p>
              <div className="w-24 h-1 bg-blue-600 mx-auto rounded-full"></div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-6xl mx-auto">
              {PERSONAS.map((persona) => (
                <button
                  key={persona.id}
                  onClick={() => selectPersona(persona)}
                  className={`p-4 rounded-lg border-2 transition-all duration-200 hover:shadow-lg hover:scale-105 text-left relative ${persona.color}`}
                >
                  {chatSessions[persona.id] && chatSessions[persona.id].length > 0 && (
                    <div className="absolute top-2 right-2 bg-green-500 text-white text-xs px-2 py-1 rounded-full font-medium">
                      {chatSessions[persona.id].length} msgs
                    </div>
                  )}
                  <div className="text-3xl mb-2">{persona.emoji}</div>
                  <h3 className="font-semibold text-lg mb-1">{persona.name}</h3>
                  <p className="text-sm opacity-80">
                    {chatSessions[persona.id] && chatSessions[persona.id].length > 0 
                      ? `${persona.description} • Continue conversation`
                      : persona.description
                    }
                  </p>
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
                      <span className="font-medium text-blue-900">
                        {messages.length > 0 ? `Welcome back, ` : `Welcome, `}
                        <b>{selectedPersona.name}</b>!
                      </span>
                      <p className="text-sm text-blue-700">
                        I'm here to help with federal student loan questions tailored to your situation.
                        {messages.length > 0 && (
                          <span className="ml-1">
                            Continuing your conversation ({messages.length} message{messages.length !== 1 ? 's' : ''}).
                          </span>
                        )}
                      </p>
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

            {/* Question Sheet - positioned after welcome banner */}
            {selectedPersona && (
              <div className="mb-4 bg-yellow-50 border border-yellow-200 shadow-sm rounded-lg overflow-hidden">
                <div className="flex items-center justify-between p-3 bg-yellow-100 border-b border-yellow-200">
                  <h3 className="text-sm font-medium text-gray-700">
                    {messages.length <= 1
                      ? (
                        <>
                          {selectedPersona.emoji} Helper questions for{' '}
                          <span className="font-bold text-gray-900">{selectedPersona.name}</span>
                        </>
                      ) : (
                        <>
                          {selectedPersona.emoji} More{' '}
                          <span className="font-bold text-gray-900">{selectedPersona.name}</span>
                          {' '}Questions
                        </>
                      )
                    }
                  </h3>
                  <button
                    onClick={() => setShowExampleQuestionsSlider(!showExampleQuestionsSlider)}
                    className="p-1 hover:bg-yellow-200 rounded-full transition-colors duration-200"
                    title={showExampleQuestionsSlider ? 'Collapse questions' : 'Expand questions'}
                  >
                    {showExampleQuestionsSlider ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </button>
                </div>
                <div
                  className={`transition-all duration-300 ease-in-out overflow-hidden ${
                    showExampleQuestionsSlider ? 'max-h-[40vh] opacity-100' : 'max-h-0 opacity-0'
                  }`}
                >
                  <div className="p-4 max-h-[40vh] overflow-y-auto">
                    <ul className="space-y-2">
                      {selectedPersona.questions.map((question, index) => (
                        <li key={index}>
                          <button
                            onClick={() => handleExampleQuestion(question)}
                            className="text-left w-full hover:bg-yellow-100 p-2 rounded transition-colors duration-200 text-sm"
                          >
                            <span className="text-blue-600">•</span>{' '}
                            <span className="font-medium text-blue-700">{question.focus}:</span>{' '}
                            <span className="text-gray-700">{question.text}</span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
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
                        <span className="flex items-center space-x-1 relative">
                          <button 
                            onClick={(e) => toggleTooltip(message.id, 'sources', e)}
                            className="cursor-pointer hover:text-blue-600 transition-colors duration-200"
                          >
                            📚 {message.sources} sources
                          </button>
                          {message.source_details && message.source_details.length > 0 && openTooltips[message.id]?.sources && (
                            <div className={`absolute left-0 w-64 bg-white border-2 border-blue-300 rounded-lg shadow-xl p-2 text-xs text-blue-900 z-[70] max-h-40 overflow-y-auto ${
                              openTooltips[message.id]?.sourcesPosition === 'below' 
                                ? 'top-full mt-2' 
                                : 'bottom-full mb-2'
                            }`}>
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
                  <div className="absolute top-2 right-2">
                    <button 
                      onClick={(e) => toggleTooltip(message.id, 'info', e)}
                      className="bg-[var(--primary)] hover:bg-[var(--primary-hover)] rounded-full p-1 transition-colors duration-200 cursor-pointer"
                    >
                      <Info className="h-3 w-3 text-white" />
                    </button>
                    {openTooltips[message.id]?.info && (
                      <div className={`absolute right-0 w-72 bg-white border-2 border-blue-300 rounded-lg shadow-xl p-3 text-xs text-blue-900 z-[60] max-h-96 overflow-y-auto transform -translate-x-1/4 ${
                        openTooltips[message.id]?.infoPosition === 'below' 
                          ? 'top-full mt-2' 
                          : 'bottom-full mb-2'
                      }`}>
                      <h3 className="font-semibold mb-2">Performance & Processing Details</h3>

                      {/* Performance Metrics */}
                      <div className="mb-2">
                        <h4 className="font-medium text-blue-900 mb-1">⏱️ Performance</h4>
                        {message.performance_metrics.response_time_ms && <p>Response Time: {(message.performance_metrics.response_time_ms / 1000).toFixed(2)}s</p>}
                        {message.performance_metrics.retrieval_time_ms && <p>Retrieval Time: {(message.performance_metrics.retrieval_time_ms / 1000).toFixed(2)}s</p>}
                        {message.performance_metrics.generation_time_ms && <p>Generation Time: {(message.performance_metrics.generation_time_ms / 1000).toFixed(2)}s</p>}
                      </div>

                      {/* Token Usage */}
                      {(message.performance_metrics.input_tokens || message.performance_metrics.output_tokens || message.performance_metrics.tokens_used) && (
                        <div className="mb-2">
                          <h4 className="font-medium text-blue-900 mb-1">🔢 Tokens</h4>
                          <p>
                            {message.performance_metrics.input_tokens && message.performance_metrics.output_tokens ? (
                              <>{message.performance_metrics.input_tokens} In | {message.performance_metrics.output_tokens} Out | 💯 {message.performance_metrics.tokens_used || (message.performance_metrics.input_tokens + message.performance_metrics.output_tokens)}</>
                            ) : message.performance_metrics.tokens_used ? (
                              <>💯 {message.performance_metrics.tokens_used} <span className="text-yellow-600 dark:text-yellow-400">(breakdown unavailable)</span></>
                            ) : null}
                          </p>
                        </div>
                      )}

                      {/* Tools Used */}
                      {message.tools_used && message.tools_used.length > 0 && (
                        <div className="mb-2">
                          <h4 className="font-medium text-blue-900 mb-1">🛠️ Tools Used</h4>
                          {message.tools_used.map((tool, index) => (
                            <p key={index}>{getHumanReadableToolName(tool)}</p>
                          ))}
                        </div>
                      )}
                      {/* Sources */}
                      {message.source_details && message.source_details.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-blue-300">
                          <h4 className="font-medium text-blue-900 mb-1">📚 Sources</h4>
                          <ul className="list-disc list-inside max-h-32 overflow-y-auto">
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
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>


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

      {/* Fixed Footer AI Disclaimer */}
      <div className="fixed bottom-0 left-0 right-0 bg-amber-50 border-t border-amber-200 px-4 py-2 text-center z-50">
        <div className="text-xs text-amber-800 font-semibold">
          🤖 AI-powered tool: For educational purposes only. Always verify with official sources.
        </div>
      </div>

      {/* Chat Input Area - Only visible when persona is selected */}
      {selectedPersona && (
        <div className="border-t border-[var(--border)] bg-[var(--background)] p-4 pb-12">
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
                onClick={clearInput}
                disabled={!inputMessage.trim() || isLoading}
                className="px-4 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                title="Clear input"
                aria-label="Clear input field"
              >
                <X className="h-4 w-4" />
              </button>
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
      )}
    </div>
  )
}