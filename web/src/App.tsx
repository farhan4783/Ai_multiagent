import React, { useState, useEffect, useRef } from 'react';
import { GraphVisualizer } from './components/GraphVisualizer';
import { GanttChart, GanttItem } from './components/GanttChart';
import { AgentSwarmMonitor, AgentInfo } from './components/AgentSwarmMonitor';
import { PromptQLConsole } from './components/PromptQLConsole';
import { InterventionModal } from './components/InterventionModal';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'graph' | 'gantt' | 'agents' | 'memory'>('graph');
  const [projectState, setProjectState] = useState<any>(null);
  const [ganttItems, setGanttItems] = useState<GanttItem[]>([]);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [telemetryLogs, setTelemetryLogs] = useState<any[]>([]);
  const [isAutonomous, setIsAutonomous] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'offline'>('connecting');
  const intervalRef = useRef<any>(null);

  // Fetch all endpoints
  const fetchAllData = async () => {
    try {
      const [stateRes, ganttRes, agentsRes] = await Promise.all([
        fetch('/api/project/state'),
        fetch('/api/project/gantt'),
        fetch('/api/agents/status'),
      ]);

      if (stateRes.ok) {
        const stateData = await stateRes.json();
        setProjectState(stateData);
      }
      if (ganttRes.ok) {
        const ganttData = await ganttRes.json();
        setGanttItems(ganttData);
      }
      if (agentsRes.ok) {
        const agentsData = await agentsRes.json();
        setAgents(agentsData);
      }
      setConnectionStatus('connected');
    } catch {
      setConnectionStatus('offline');
      // Populate rich default sample state if running standalone UI
      if (!projectState) {
        setProjectState({
          project_id: 'stgat-marl-system',
          current_step: 3,
          makespan: 46.0,
          critical_path: ['TASK-01', 'TASK-04', 'TASK-05', 'TASK-06'],
          total_slack: 6.0,
          risk_index: 1.77,
          tasks: {
            'TASK-01': { task_id: 'TASK-01', title: 'Design Core Schemas & Pydantic Models', status: 'completed', estimated_duration: 8.0, is_critical: true, priority: 5 },
            'TASK-02': { task_id: 'TASK-02', title: 'Tree-Sitter AST & Complexity Analyzer', status: 'completed', estimated_duration: 12.0, is_critical: false, priority: 4 },
            'TASK-03': { task_id: 'TASK-03', title: 'Git Commit Co-Churn History Miner', status: 'completed', estimated_duration: 10.0, is_critical: false, priority: 3 },
            'TASK-04': { task_id: 'TASK-04', title: 'AON DAG Builder & CPM Engine', status: 'in_progress', estimated_duration: 14.0, is_critical: true, priority: 5, assigned_agent_id: 'developer_agent_0' },
            'TASK-05': { task_id: 'TASK-05', title: 'Synthesize Unified Spatiotemporal Graph', status: 'pending', estimated_duration: 16.0, is_critical: true, priority: 5 },
            'TASK-06': { task_id: 'TASK-06', title: 'End-to-End Verification Test Suite', status: 'pending', estimated_duration: 8.0, is_critical: true, priority: 4 },
          },
          modules: {
            'src/models.py': { lines_of_code: 320, cyclomatic_complexity: 4.2 },
            'src/auth.py': { lines_of_code: 450, cyclomatic_complexity: 8.5 },
            'src/builder.py': { lines_of_code: 280, cyclomatic_complexity: 3.1 },
            'src/env.py': { lines_of_code: 600, cyclomatic_complexity: 12.0 },
          },
          edges: [
            { source: 'TASK-01', target: 'TASK-04', edge_type: 'explicit_precedence', attention_weight: 0.85 },
            { source: 'TASK-04', target: 'TASK-05', edge_type: 'explicit_precedence', attention_weight: 0.92 },
            { source: 'TASK-05', target: 'TASK-06', edge_type: 'explicit_precedence', attention_weight: 0.45 },
            { source: 'src/models.py', target: 'src/auth.py', edge_type: 'implicit_ast_call', attention_weight: 0.65 },
          ],
          predictions: { predicted_makespan_hrs: 44.8, avg_fragility: 1.42 },
          active_dispatches: { 'TASK-04': 'developer_agent_0' },
        });

        setGanttItems([
          { task_id: 'TASK-01', title: 'Design Core Schemas & Pydantic Models', start: 0, duration: 8, end: 8, late_start: 0, late_finish: 8, total_slack: 0, status: 'completed', is_critical: true, priority: 5 },
          { task_id: 'TASK-02', title: 'Tree-Sitter AST & Complexity Analyzer', start: 8, duration: 12, end: 20, late_start: 14, late_finish: 26, total_slack: 6, status: 'completed', is_critical: false, priority: 4 },
          { task_id: 'TASK-03', title: 'Git Commit Co-Churn History Miner', start: 8, duration: 10, end: 18, late_start: 16, late_finish: 26, total_slack: 8, status: 'completed', is_critical: false, priority: 3 },
          { task_id: 'TASK-04', title: 'AON DAG Builder & CPM Engine', start: 8, duration: 14, end: 22, late_start: 8, late_finish: 22, total_slack: 0, status: 'in_progress', is_critical: true, priority: 5, assigned_agent_id: 'developer_agent_0' },
          { task_id: 'TASK-05', title: 'Synthesize Unified Spatiotemporal Graph', start: 22, duration: 16, end: 38, late_start: 22, late_finish: 38, total_slack: 0, status: 'pending', is_critical: true, priority: 5 },
          { task_id: 'TASK-06', title: 'End-to-End Verification Test Suite', start: 38, duration: 8, end: 46, late_start: 38, late_finish: 46, total_slack: 0, status: 'pending', is_critical: true, priority: 4 },
        ]);

        setAgents([
          { agent_id: 'scheduling_agent', role: 'scheduling_agent', is_idle: false, current_task_id: null, allocated_port: null, worktree_path: null, health_score: 1.0, skills: ['dispatch', 'cpm'] },
          { agent_id: 'risk_agent', role: 'risk_agent', is_idle: false, current_task_id: null, allocated_port: null, worktree_path: null, health_score: 1.0, skills: ['buffer_injection', 'preflight_test'] },
          { agent_id: 'pm_agent', role: 'pm_coordinator', is_idle: false, current_task_id: null, allocated_port: null, worktree_path: null, health_score: 1.0, skills: ['reward_arbitration'] },
          { agent_id: 'developer_agent_0', role: 'developer_agent', is_idle: false, current_task_id: 'TASK-04', allocated_port: 24000, worktree_path: '.worktrees/agent-0-task-4', health_score: 0.98, skills: ['python', 'ast'] },
          { agent_id: 'developer_agent_1', role: 'developer_agent', is_idle: true, current_task_id: null, allocated_port: null, worktree_path: null, health_score: 1.0, skills: ['python', 'testing'] },
          { agent_id: 'developer_agent_2', role: 'developer_agent', is_idle: true, current_task_id: null, allocated_port: null, worktree_path: null, health_score: 1.0, skills: ['python', 'git'] },
        ]);
      }
    }
  };

  // Connect WebSocket for live streaming with auto-reconnection
  useEffect(() => {
    fetchAllData();

    let ws: WebSocket | null = null;
    let reconnectTimer: any = null;
    let isDisposed = false;

    const connectWs = () => {
      if (isDisposed) return;
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          if (!isDisposed) setConnectionStatus('connected');
        };

        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            setTelemetryLogs((prev) => [...prev.slice(-30), msg]);
            if (msg.event_type === 'INITIAL_STATE' || msg.event_type === 'STEP_COMPLETED') {
              fetchAllData();
            }
          } catch {
            // ignore parse errors
          }
        };

        ws.onclose = () => {
          if (!isDisposed) {
            setConnectionStatus('offline');
            reconnectTimer = setTimeout(connectWs, 3000);
          }
        };

        ws.onerror = () => {
          if (!isDisposed) setConnectionStatus('offline');
        };
      } catch {
        if (!isDisposed) {
          setConnectionStatus('offline');
          reconnectTimer = setTimeout(connectWs, 3000);
        }
      }
    };

    connectWs();

    return () => {
      isDisposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) ws.close();
    };
  }, []);

  // Autonomous run loop
  useEffect(() => {
    if (isAutonomous) {
      intervalRef.current = setInterval(async () => {
        try {
          await fetch('/api/orchestrator/step', { method: 'POST' });
          fetchAllData();
        } catch {
          // fallback advance step
          setProjectState((prev: any) => ({
            ...prev,
            current_step: (prev?.current_step || 0) + 1,
          }));
        }
      }, 3000);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isAutonomous]);

  // Step trigger handler
  const handleSingleStep = async () => {
    try {
      await fetch('/api/orchestrator/step', { method: 'POST' });
      fetchAllData();
    } catch {
      // simulate in standalone mode
      setProjectState((prev: any) => ({
        ...prev,
        current_step: (prev?.current_step || 0) + 1,
      }));
    }
  };

  // Reset handler
  const handleReset = async () => {
    if (!window.confirm('Reset orchestrator state and teardown all parallel worktrees?')) return;
    try {
      await fetch('/api/orchestrator/reset', { method: 'POST' });
      setIsAutonomous(false);
      fetchAllData();
    } catch {
      setIsAutonomous(false);
    }
  };

  const tasksCount = Object.keys(projectState?.tasks || {}).length;
  const completedCount = Object.values(projectState?.tasks || {}).filter((t: any) => t.status === 'completed').length;
  const activeWorktrees = Object.keys(projectState?.active_dispatches || {}).length;
  const makespan = projectState?.makespan || 46.0;
  const predictedMakespan = projectState?.predictions?.predicted_makespan_hrs || makespan;
  const riskIndex = projectState?.risk_index || 1.77;

  return (
    <div>
      {/* Header Bar */}
      <header className="app-header">
        <div className="brand-wrapper">
          <div className="brand-logo">⚡</div>
          <div>
            <div className="brand-title">STGAT-MARL ORCHESTRATOR</div>
            <div className="brand-subtitle">Autonomous Swarm Project Management OS</div>
          </div>
        </div>

        <div className="header-controls">
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem',
              fontSize: '0.75rem',
              padding: '0.3rem 0.6rem',
              borderRadius: '9999px',
              background: connectionStatus === 'connected' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
              color: connectionStatus === 'connected' ? '#10b981' : '#ef4444',
              border: `1px solid ${connectionStatus === 'connected' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
            }}
          >
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: connectionStatus === 'connected' ? '#10b981' : '#ef4444' }} />
            {connectionStatus.toUpperCase()}
          </span>

          <button
            id="step-dispatch-btn"
            className="btn btn-secondary"
            onClick={handleSingleStep}
          >
            ▶ Step
          </button>

          <button
            id="autonomous-toggle-btn"
            className={`btn ${isAutonomous ? 'btn-danger' : 'btn-primary'}`}
            onClick={() => setIsAutonomous(!isAutonomous)}
          >
            {isAutonomous ? '⏸ Pause Swarm' : '⚡ Auto-Dispatch'}
          </button>

          <button
            id="intervention-modal-btn"
            className="btn btn-secondary"
            onClick={() => setIsModalOpen(true)}
          >
            ⚙ Intervene
          </button>

          <button
            id="reset-orchestrator-btn"
            className="btn btn-secondary"
            onClick={handleReset}
          >
            ↺ Reset
          </button>
        </div>
      </header>

      {/* Top Metrics Cards */}
      <section className="metrics-grid">
        <div className="metric-card glass-panel">
          <span className="metric-label">Predicted Makespan</span>
          <div className="metric-value">{predictedMakespan}h</div>
          <span className="metric-sub">CPM Baseline: {makespan}h</span>
        </div>

        <div className="metric-card glass-panel purple">
          <span className="metric-label">Structural Risk Index</span>
          <div className="metric-value">{riskIndex.toFixed(2)}</div>
          <span className="metric-sub">STGAT Attention Weighted</span>
        </div>

        <div className="metric-card glass-panel green">
          <span className="metric-label">Active Worktrees</span>
          <div className="metric-value">{activeWorktrees} / 3</div>
          <span className="metric-sub">Zero Port/DB Collisions</span>
        </div>

        <div className="metric-card glass-panel amber">
          <span className="metric-label">Tasks Progress</span>
          <div className="metric-value">{completedCount} / {tasksCount}</div>
          <span className="metric-sub">Critical Path: {projectState?.critical_path?.length || 0} tasks</span>
        </div>

        <div className="metric-card glass-panel">
          <span className="metric-label">Orchestration Cycle</span>
          <div className="metric-value">Step {projectState?.current_step || 1}</div>
          <span className="metric-sub">Total Float: {projectState?.total_slack?.toFixed(1) || '0.0'}h</span>
        </div>
      </section>

      {/* View Tabs */}
      <nav className="tabs-nav">
        <button
          id="tab-graph-btn"
          className={`tab-btn ${activeTab === 'graph' ? 'active' : ''}`}
          onClick={() => setActiveTab('graph')}
        >
          Spatiotemporal Graph & Attention Heatmap
        </button>
        <button
          id="tab-gantt-btn"
          className={`tab-btn ${activeTab === 'gantt' ? 'active' : ''}`}
          onClick={() => setActiveTab('gantt')}
        >
          Dynamic Gantt Timeline
        </button>
        <button
          id="tab-agents-btn"
          className={`tab-btn ${activeTab === 'agents' ? 'active' : ''}`}
          onClick={() => setActiveTab('agents')}
        >
          Agent Swarm & Sandboxed Worktrees
        </button>
        <button
          id="tab-memory-btn"
          className={`tab-btn ${activeTab === 'memory' ? 'active' : ''}`}
          onClick={() => setActiveTab('memory')}
        >
          PromptQL Shared Context & RAG Memory
        </button>
      </nav>

      {/* Main Content Body */}
      <main className="dashboard-content">
        {activeTab === 'graph' && (
          <GraphVisualizer
            tasks={projectState?.tasks || {}}
            modules={projectState?.modules || {}}
            edges={projectState?.edges || []}
            predictions={projectState?.predictions || {}}
          />
        )}

        {activeTab === 'gantt' && (
          <GanttChart
            items={ganttItems}
            totalMakespan={makespan}
          />
        )}

        {activeTab === 'agents' && (
          <AgentSwarmMonitor
            agents={agents}
            telemetryLogs={telemetryLogs}
          />
        )}

        {activeTab === 'memory' && (
          <PromptQLConsole />
        )}
      </main>

      {/* Human-in-the-Loop Intervention Modal */}
      <InterventionModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        tasks={projectState?.tasks || {}}
        onApplied={() => {
          fetchAllData();
        }}
      />
    </div>
  );
};
