import React from 'react';

export interface GanttItem {
  task_id: string;
  title: string;
  start: number;
  duration: number;
  end: number;
  late_start: number;
  late_finish: number;
  total_slack: number;
  status: string;
  is_critical: boolean;
  assigned_agent_id?: string;
  priority: number;
}

interface GanttChartProps {
  items: GanttItem[];
  totalMakespan: number;
}

export const GanttChart: React.FC<GanttChartProps> = ({ items, totalMakespan }) => {
  const maxHorizon = Math.max(totalMakespan, 20.0);
  const timeTicks = [0, 0.25, 0.5, 0.75, 1.0].map((pct) => Math.round(pct * maxHorizon));

  return (
    <div className="gantt-container glass-panel" id="gantt-chart-panel">
      {/* Time Axis Header */}
      <div style={{ display: 'flex', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '0.5rem', marginBottom: '0.75rem' }}>
        <div style={{ width: '240px', flexShrink: 0, fontSize: '0.75rem', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase' }}>
          Task Item
        </div>
        <div style={{ flexGrow: 1, display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace', color: '#64748b' }}>
          {timeTicks.map((tick) => (
            <span key={tick}>{tick}h</span>
          ))}
        </div>
      </div>

      {/* Task Gantt Rows */}
      {items.length === 0 ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: '#64748b', fontSize: '0.85rem' }}>
          No tasks in backlog. Ingest a project specification to generate dynamic Gantt schedule.
        </div>
      ) : (
        items.map((item) => {
          const leftPct = (item.start / maxHorizon) * 100;
          const widthPct = Math.max((item.duration / maxHorizon) * 100, 2.5);
          const slackWidthPct = (item.total_slack / maxHorizon) * 100;

          let barClass = 'normal';
          if (item.status === 'completed') {
            barClass = 'completed';
          } else if (item.is_critical) {
            barClass = 'critical';
          }

          return (
            <div className="gantt-row" key={item.task_id}>
              <div className="gantt-label">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span className="gantt-task-id">{item.task_id}</span>
                  {item.assigned_agent_id && (
                    <span style={{ fontSize: '0.65rem', background: 'rgba(0, 240, 255, 0.15)', color: '#00f0ff', padding: '0.1rem 0.35rem', borderRadius: '4px' }}>
                      {item.assigned_agent_id.replace('developer_agent_', 'dev-')}
                    </span>
                  )}
                </div>
                <span className="gantt-task-title" title={item.title}>{item.title}</span>
              </div>

              <div className="gantt-timeline-bar-area">
                {/* Total Slack Float Bar (Dashed amber/slate) */}
                {item.total_slack > 0 && item.status !== 'completed' && (
                  <div
                    style={{
                      position: 'absolute',
                      left: `${leftPct + widthPct}%`,
                      width: `${slackWidthPct}%`,
                      height: '100%',
                      background: 'repeating-linear-gradient(45deg, transparent, transparent 5px, rgba(245, 158, 11, 0.15) 5px, rgba(245, 158, 11, 0.15) 10px)',
                      border: '1px dashed rgba(245, 158, 11, 0.4)',
                      borderRadius: '4px',
                    }}
                    title={`Float / Slack: ${item.total_slack.toFixed(1)} hrs`}
                  />
                )}

                {/* Primary Task Duration Bar */}
                <div
                  className={`gantt-bar ${barClass}`}
                  style={{
                    left: `${leftPct}%`,
                    width: `${widthPct}%`,
                  }}
                  title={`Start: ${item.start.toFixed(1)}h | Duration: ${item.duration.toFixed(1)}h | End: ${item.end.toFixed(1)}h`}
                >
                  <span>{item.duration.toFixed(1)}h</span>
                </div>
              </div>
            </div>
          );
        })
      )}
    </div>
  );
};
