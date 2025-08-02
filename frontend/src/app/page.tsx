'use client'

import { useState } from 'react'
import { Send, MessageSquare, Clock, CheckCircle, AlertCircle } from 'lucide-react'

interface Message {
  id: string
  content: string
  isUser: boolean
  timestamp: Date
  sources?: number
  processingTime?: string
}

interface ApiResponse {
  answer: string
  sources_count: number
  success: boolean
  message?: string
  performance_metrics?: {
    total_processing_time: number
  }
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your Federal Student Loan Assistant. I can help answer questions about federal student loan policies, procedures, and common issues. What would you like to know?',
      isUser: false,
      timestamp: new Date()
    }
  ])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState<'unknown' | 'healthy' | 'unhealthy'>('unknown')

  // Check API health on component mount
  useState(() => {
    checkApiHealth()
  })

  const checkApiHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/health')
      const data = await response.json()
      setApiStatus(data.status === 'healthy' ? 'healthy' : 'unhealthy')
    } catch (error) {
      setApiStatus('unhealthy')
    }
  }

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

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
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: ApiResponse = await response.json()

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.answer,
        isUser: false,
        timestamp: new Date(),
        sources: data.sources_count,
        processingTime: data.performance_metrics?.total_processing_time 
          ? `${data.performance_metrics.total_processing_time.toFixed(2)}s`
          : undefined
      }

      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, I encountered an error while processing your question. Please make sure the backend API is running on localhost:8000 and try again.',
        isUser: false,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <MessageSquare className="h-8 w-8" />
            <h1 className="text-xl font-semibold">Federal Student Loan Assistant</h1>
          </div>
          <div className="flex items-center space-x-2">
            {apiStatus === 'healthy' && (
              <div className="flex items-center space-x-1 text-green-200">
                <CheckCircle className="h-4 w-4" />
                <span className="text-sm">Online</span>
              </div>
            )}
            {apiStatus === 'unhealthy' && (
              <div className="flex items-center space-x-1 text-red-200">
                <AlertCircle className="h-4 w-4" />
                <span className="text-sm">Offline</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Chat Messages */}
      <div className="flex-1 max-w-4xl mx-auto w-full p-4">
        <div className="space-y-4 mb-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl rounded-lg p-4 ${
                  message.isUser
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-800 shadow-md border'
                }`}
              >
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
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white text-gray-800 shadow-md border rounded-lg p-4 max-w-3xl">
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  <span>Searching through federal student loan knowledge base...</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t bg-white p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex space-x-4">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me anything about federal student loans..."
              className="flex-1 border border-gray-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={2}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              <Send className="h-4 w-4" />
              <span>Send</span>
            </button>
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Press Enter to send, Shift+Enter for new line
          </div>
        </div>
      </div>
    </div>
  )
}