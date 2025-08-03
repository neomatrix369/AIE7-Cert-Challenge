'use client'

import { useState, useEffect, useRef } from 'react'
import { Send, MessageSquare, Clock, CheckCircle, AlertCircle, Zap, Info } from 'lucide-react'

interface PerformanceMetrics {
  response_time_ms?: number;
  retrieval_time_ms?: number;
  generation_time_ms?: number;
  confidence_score?: number;
  tokens_used?: number;
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
}

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
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages]);

  useEffect(() => {
    checkApiHealth()
    setMessages([
      {
        id: '1',
        content: 'Hello! I\'m your Federal Student Loan Assistant. I can help answer questions about federal student loan policies, procedures, and common issues. What would you like to know?',
        isUser: false,
        timestamp: new Date()
      }
    ])
  }, [])

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
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      console.log('Frontend: Sending request to backend /ask endpoint')
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: inputMessage,
          max_response_length: 2000
        })
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

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleExampleQuestion = (question: string) => {
    setInputMessage(question)
  }

  return (
    <div className="min-h-screen bg-[var(--background-light)] dark:bg-[var(--background-dark)] flex flex-col">
      <header className="bg-[var(--primary-light)] dark:bg-[var(--primary-dark)] text-white p-4 shadow-md">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <MessageSquare className="h-8 w-8" />
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

      <div className="flex-1 max-w-4xl mx-auto w-full p-4">
        <div className="space-y-4 mb-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`relative max-w-3xl rounded-lg p-4 shadow-md ${message.isUser ? 'bg-[var(--user-message-bg-light)] dark:bg-[var(--user-message-bg-dark)] text-white' : 'bg-[var(--bot-message-bg-light)] dark:bg-[var(--bot-message-bg-dark)] text-[var(--foreground-light)] dark:text-[var(--foreground-dark)] border border-[var(--border-light)] dark:border-[var(--border-dark)]'}`}>
                <div className="whitespace-pre-wrap">{message.content}</div>
                <div className="flex items-center justify-between mt-2 text-xs opacity-70">
                  <span>{message.timestamp.toLocaleTimeString()}</span>
                  {!message.isUser && (
                    <div className="flex items-center space-x-3">
                      {message.sources && (
                        <span className="flex items-center space-x-1">
                          <span>📚 {message.sources} sources</span>
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
                    <Info className="h-4 w-4 text-gray-400 cursor-pointer" />
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
                              <li key={index} className="truncate">{typeof source === 'string' ? source : JSON.stringify(source)}</li>
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

        {messages.length <= 1 && !isLoading && (
          <div className="text-center text-gray-500 dark:text-gray-400">
            <h2 className="text-lg font-semibold mb-2">Try asking a question!</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <button onClick={() => handleExampleQuestion('What is the difference between a subsidized and unsubsidized loan?')} className="bg-gray-200 dark:bg-gray-700 p-2 rounded-lg text-sm">What is the difference between a subsidized and unsubsidized loan?</button>
              <button onClick={() => handleExampleQuestion('How do I apply for a Direct PLUS Loan?')} className="bg-gray-200 dark:bg-gray-700 p-2 rounded-lg text-sm">How do I apply for a Direct PLUS Loan?</button>
              <button onClick={() => handleExampleQuestion('What are the current interest rates for federal student loans?')} className="bg-gray-200 dark:bg-gray-700 p-2 rounded-lg text-sm">What are the current interest rates for federal student loans?</button>
              <button onClick={() => handleExampleQuestion('Can I consolidate my federal student loans?')} className="bg-gray-200 dark:bg-gray-700 p-2 rounded-lg text-sm">Can I consolidate my federal student loans?</button>
            </div>
          </div>
        )}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-[var(--bot-message-bg-light)] dark:bg-[var(--bot-message-bg-dark)] text-[var(--foreground-light)] dark:text-[var(--foreground-dark)] shadow-md border border-[var(--border-light)] dark:border-[var(--border-dark)] rounded-lg p-4 max-w-3xl">
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-[var(--primary-light)] dark:border-[var(--primary-dark)]"></div>
                <span>Searching through federal student loan knowledge base...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="border-t border-[var(--border-light)] dark:border-[var(--border-dark)] bg-[var(--background-light)] dark:bg-[var(--background-dark)] p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex space-x-4">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me anything about federal student loans..."
              className="flex-1 border border-gray-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-[var(--primary-light)] dark:focus:ring-[var(--primary-dark)] focus:border-transparent resize-none bg-white dark:bg-gray-800 text-black dark:text-white"
              rows={2}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="px-6 py-3 bg-[var(--primary-light)] dark:bg-[var(--primary-dark)] text-white rounded-lg hover:bg-[var(--primary-hover-light)] dark:hover:bg-[var(--primary-hover-dark)] focus:outline-none focus:ring-2 focus:ring-[var(--primary-light)] dark:focus:ring-[var(--primary-dark)] disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2">
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