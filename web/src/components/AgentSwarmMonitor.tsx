import React from 'react';

export interface AgentInfo {
  agent_id: string;
  role: string;
  is_idle: boolean;
  current_task_id: string | null;
  allocated_port: number | null;
  worktree_path: string | null;
  health_score: number;
  skills: string[];
}

interface AgentSwarmMonitorProps {
  agents: AgentInfo[];
  telemetryLogs: Array<any>;
}

export const AgentSwarmMonitor: React.FC<AgentSwarmMonitorProps> = ({
  agents,
  telemetryLogs,
}) => {
  return (
    <div className="agents-grid" id="agent-swarm-monitor-grid">
      {agents.map((agent) => {
        const isIdle = agent.is_idle;
        const isDev = agent.role === 'developer_agent';
        const roleLabel = agent.role.replace('_agent', '').replace('_', ' ').toUpperCase();

        // Filter recent logs for this agent
        const relevantLogs = telemetryLogs
          .filter((log) => {
            const data = log.data || {};
            if (data.agent_id === agent.agent_id) return true;
            if (Array.isArray(data.dispatches)) {
              return data.dispatches.some((d: any) => d.agent_id === agent.agent_id);
            }
            return false;
          })
          .slice(-6);

        return (
          <div className="agent-card glass-panel" key={agent.agent_id}>
            <div className="agent-card-header">
              <div>
                <div className="agent-name">{agent.agent_id.replace('_', ' ').toUpperCase()}</div>
                <div style={{ fontSize: '0.7rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  {roleLabel} ROLE
                </div>
              </div>
              <span className={`agent-badge ${isIdle ? 'idle' : 'busy'}`}>
                {isIdle ? 'IDLE' : 'EXECUTING'}
              </span>
            </div>

            {/* Agent Properties */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.75rem' }}>
              <div>
                <span style={{ color: '#64748b' }}>Current Task:</span>{' '}
                <strong style={{ color: agent.current_task_id ? '#00f0ff' : '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>
                  {agent.current_task_id || 'None'}
                </strong>
              </div>
              <div>
                <span style={{ color: '#64748b' }}>Health:</span>{' '}
                <strong style={{ color: '#10b981' }}>{(agent.health_score * 100).toFixed(0)}%</strong>
              </div>

              {isDev && (
                <>
                  <div>
                    <span style={{ color: '#64748b' }}>Port Lease:</span>{' '}
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', color: agent.allocated_port ? '#f59e0b' : '#64748b' }}>
                      {agent.allocated_port ? `:${agent.allocated_port}` : 'Unassigned'}
                    </span>
                  </div>
                  <div>
                    <span style={{ color: '#64748b' }}>Worktree:</span>{' '}
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.7rem', color: agent.worktree_path ? '#a855f7' : '#64748b' }}>
                      {agent.worktree_path ? '.worktree/active' : 'Clean'}
                    </span>
                  </div>
                </>
              )}
            </div>

            {/* Terminal Live Output Log Stream */}
            <div className="terminal-box">
              <div style={{ color: '#64748b', marginBottom: '0.3rem', fontSize: '0.7rem' }}>
                $ mcp-runner --agent={agent.agent_id} telemetry
              </div>
              {relevantLogs.length === 0 ? (
                <div style={{ color: '#475569', fontStyle: 'italic' }}>Awaiting agent dispatch events...</div>
              ) : (
                relevantLogs.map((log, idx) => (
                  <div key={idx} style={{ marginBottom: '0.2rem' }}>
                    <span style={{ color: '#64748b' }}>[{new Date(log.timestamp * 1000).toLocaleTimeString()}]</span>{' '}
                    <span style={{ color: '#38bdf8' }}>{log.event_type}:</span>{' '}
                    <span style={{ color: '#cbd5e1' }}>{JSON.stringify(log.data).slice(0, 70)}...</span>
                  </div>
                ))
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};
