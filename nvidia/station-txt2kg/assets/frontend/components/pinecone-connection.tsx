"use client"

import { useState, useEffect } from "react"
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { InfoIcon } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { VectorDBStats } from '@/types/graph'

interface GeodeVectorConnectionProps {
  className?: string
}

export function GeodeVectorConnection({ className }: GeodeVectorConnectionProps) {
  const [connectionStatus, setConnectionStatus] = useState<"connected" | "disconnected" | "checking">("disconnected")
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState<VectorDBStats>({ nodes: 0, relationships: 0, source: 'none' })

  // Fetch vector DB stats from Geode
  const fetchStats = async () => {
    try {
      const response = await fetch('/api/vector-db/stats');
      const data = await response.json();

      if (response.ok) {
        setStats({
          nodes: typeof data.totalVectorCount === 'number' ? data.totalVectorCount : 0,
          relationships: 0, // Vector embeddings don't store relationships directly
          source: data.source || 'unknown',
          httpHealthy: data.httpHealthy
        });

        // If we have a healthy connection, we're connected
        if (data.httpHealthy) {
          setConnectionStatus("connected");
          setError(null);
        } else {
          setConnectionStatus("disconnected");
          setError(data.error || 'Connection failed');
        }

        console.log('Geode vector stats:', data);
      } else {
        console.error('Failed to fetch vector stats:', data);
        setConnectionStatus("disconnected");
        setError(data.error || 'Failed to connect to Geode vector store');
      }
    } catch (error) {
      console.error('Error fetching vector stats:', error);
      setConnectionStatus("disconnected");
      setError(error instanceof Error ? error.message : 'Error connecting to Geode vector store');
    }
  };

  // Check connection status and stats
  const checkConnection = async () => {
    setConnectionStatus("checking")
    setError(null)

    try {
      await fetchStats();
    } catch (error) {
      console.error('Error connecting to Geode vector store:', error)
      setConnectionStatus("disconnected")
      setError(error instanceof Error ? error.message : 'Unknown error connecting to Geode vector store')
    }
  }

  // Reset connection state
  const disconnect = async () => {
    setConnectionStatus("disconnected")
    setStats({ nodes: 0, relationships: 0, source: 'none' })
  }

  // Initial connection check
  useEffect(() => {
    checkConnection()
  }, [])

  return (
    <div className={`flex flex-col items-start space-y-4 p-4 border rounded-md ${className}`}>
      <div className="flex justify-between w-full">
        <h2 className="text-lg font-medium">Vector DB</h2>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger>
              <InfoIcon className="h-5 w-5 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent>
              <p>Geode stores vector embeddings for semantic search alongside graph data</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      <div className="flex items-center space-x-2">
        <span className="text-sm">Status:</span>
        {connectionStatus === "connected" ? (
          <Badge variant="outline" className="bg-green-50 text-green-700 hover:bg-green-50 border-green-200">Connected</Badge>
        ) : connectionStatus === "checking" ? (
          <Badge variant="outline" className="bg-yellow-50 text-yellow-700 hover:bg-yellow-50 border-yellow-200">Checking...</Badge>
        ) : (
          <Badge variant="outline" className="bg-red-50 text-red-700 hover:bg-red-50 border-red-200">Disconnected</Badge>
        )}
      </div>

      {error && (
        <div className="text-sm text-red-600 bg-red-50 p-2 rounded w-full overflow-auto max-h-20">
          <p className="whitespace-normal break-words">Error: {error}</p>
          {error.includes('404') && (
            <p className="mt-1 text-xs">
              The Geode server is running but the collection doesn't exist yet.
              <button
                onClick={async () => {
                  setConnectionStatus("checking");
                  setError(null);
                  try {
                    const response = await fetch('/api/vector-db/create-collection', { method: 'POST' });
                    if (response.ok) {
                      // Wait a bit for the collection to be created
                      await new Promise(resolve => setTimeout(resolve, 2000));
                      checkConnection();
                    } else {
                      const data = await response.json();
                      setError(data.error || 'Failed to create collection');
                      setConnectionStatus("disconnected");
                    }
                  } catch (err) {
                    setError(err instanceof Error ? err.message : 'Error creating collection');
                    setConnectionStatus("disconnected");
                  }
                }}
                className="ml-1 text-blue-600 hover:text-blue-800 underline"
              >
                Click here to create the collection
              </button>
            </p>
          )}
        </div>
      )}

      <div className="text-sm space-y-1 w-full">
        <div className="flex justify-between">
          <span className="text-muted-foreground">Vectors:</span>
          <span>{stats.nodes}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Source:</span>
          <span>{stats.source}</span>
        </div>
      </div>

      <div className="flex space-x-2">
        <Button
          variant="outline"
          size="sm"
          onClick={checkConnection}
          disabled={connectionStatus === "checking"}
        >
          {connectionStatus === "checking" ? "Checking..." : "Check Connection"}
        </Button>

        {connectionStatus === "connected" && (
          <Button
            variant="outline"
            size="sm"
            onClick={disconnect}
          >
            Disconnect
          </Button>
        )}
      </div>
    </div>
  )
}
