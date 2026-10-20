import React, { useEffect, useRef, useState } from 'react';

interface GraphVisualizerProps {
  tasks: Record<string, any>;
  modules: Record<string, any>;
  edges: Array<any>;
  predictions: Record<string, any>;
  onSelectNode?: (nodeId: string, nodeType: 'task' | 'module') => void;
}

export const GraphVisualizer: React.FC<GraphVisualizerProps> = ({
  tasks,
  modules,
  edges,
  predictions,
  onSelectNode,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Adjust canvas resolution for retina displays
    const width = canvas.parentElement?.clientWidth || 800;
    const height = 540;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const taskKeys = Object.keys(tasks);
    const moduleKeys = Object.keys(modules);
    const totalNodes = taskKeys.length + moduleKeys.length;

    if (totalNodes === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '14px Inter, sans-serif';
      ctx.fillText('No graph nodes loaded.', 20, 40);
      return;
    }

    // Node positioning in dual concentric / bipartite layout
    const nodePositions: Record<string, { x: number; y: number; type: 'task' | 'module' }> = {};

    // Center task nodes in an inner ellipse
    const centerX = width / 2;
    const centerY = height / 2;
    const taskRadiusX = Math.min(width * 0.28, 220);
    const taskRadiusY = Math.min(height * 0.32, 140);

    taskKeys.forEach((tid, i) => {
      const angle = (i / Math.max(taskKeys.length, 1)) * 2 * Math.PI;
      nodePositions[tid] = {
        x: centerX + taskRadiusX * Math.cos(angle),
        y: centerY + taskRadiusY * Math.sin(angle),
        type: 'task',
      };
    });

    // Outer circle for module nodes
    const modRadiusX = Math.min(width * 0.44, 340);
    const modRadiusY = Math.min(height * 0.42, 210);

    moduleKeys.forEach((mid, i) => {
      const angle = (i / Math.max(moduleKeys.length, 1)) * 2 * Math.PI + 0.2;
      nodePositions[mid] = {
        x: centerX + modRadiusX * Math.cos(angle),
        y: centerY + modRadiusY * Math.sin(angle),
        type: 'module',
      };
    });

    let animationFrameId: number;
    let pulseTime = 0;

    const render = () => {
      pulseTime += 0.04;
      ctx.clearRect(0, 0, width, height);

      // 1. Draw Edges
      edges.forEach((edge) => {
        const src = nodePositions[edge.source];
        const tgt = nodePositions[edge.target];
        if (!src || !tgt) return;

        const attnWeight = edge.attention_weight ?? 0.5;
        const isHighRisk = attnWeight > 0.7;

        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);

        if (isHighRisk) {
          // Dynamic pulsing crimson/amber attention edge
          const pulse = (Math.sin(pulseTime * 3) + 1) / 2;
          ctx.strokeStyle = `rgba(239, 68, 68, ${0.5 + 0.5 * pulse})`;
          ctx.lineWidth = 2.5 + pulse * 1.5;
        } else if (edge.edge_type === 'explicit_precedence') {
          ctx.strokeStyle = 'rgba(0, 240, 255, 0.45)';
          ctx.lineWidth = 1.8;
        } else if (edge.edge_type === 'implicit_ast_call') {
          ctx.strokeStyle = 'rgba(168, 85, 247, 0.35)';
          ctx.lineWidth = 1.2;
        } else {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
          ctx.lineWidth = 1.0;
        }

        ctx.stroke();

        // Draw arrow for directed edge
        const midX = (src.x + tgt.x) / 2;
        const midY = (src.y + tgt.y) / 2;
        const angle = Math.atan2(tgt.y - src.y, tgt.x - src.x);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath();
        ctx.arc(midX, midY, 2.5, 0, 2 * Math.PI);
        ctx.fill();
      });

      // 2. Draw Module Nodes (Circles)
      moduleKeys.forEach((mid) => {
        const pos = nodePositions[mid];
        const mod = modules[mid];

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 9, 0, 2 * Math.PI);
        ctx.fillStyle = '#3b82f6';
        ctx.fill();
        ctx.strokeStyle = '#60a5fa';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Label
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px JetBrains Mono, monospace';
        const label = mid.split('/').pop() || mid;
        ctx.fillText(label.slice(0, 14), pos.x + 12, pos.y + 3);
      });

      // 3. Draw Task Nodes (Hexagon/Rect)
      taskKeys.forEach((tid) => {
        const pos = nodePositions[tid];
        const task = tasks[tid];
        const isCritical = task?.is_critical;
        const status = task?.status || 'pending';

        // Status color mapping
        let fillColor = '#0f172a';
        let strokeColor = '#00f0ff';
        if (status === 'completed') {
          strokeColor = '#10b981';
        } else if (status === 'in_progress') {
          strokeColor = '#00f0ff';
          fillColor = 'rgba(0, 240, 255, 0.15)';
        } else if (status === 'failed') {
          strokeColor = '#ef4444';
        }

        if (isCritical) {
          strokeColor = '#f59e0b';
        }

        const size = 32;
        ctx.save();
        ctx.translate(pos.x, pos.y);

        // Critical pulse aura
        if (isCritical) {
          const auraPulse = (Math.sin(pulseTime * 2) + 1) / 2;
          ctx.beginPath();
          ctx.arc(0, 0, size * 0.8 + auraPulse * 4, 0, 2 * Math.PI);
          ctx.fillStyle = 'rgba(245, 158, 11, 0.2)';
          ctx.fill();
        }

        // Rounded Task Card Box
        ctx.beginPath();
        ctx.roundRect(-size / 2, -size / 2, size, size, 6);
        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = isCritical ? 2.5 : 1.8;
        ctx.stroke();

        // Task ID Label
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 9px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(tid, 0, 0);

        ctx.restore();
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [tasks, modules, edges, predictions]);

  return (
    <div className="graph-canvas-container glass-panel" id="graph-visualizer-panel">
      <canvas ref={canvasRef} id="graph-canvas" />
      <div className="canvas-legend">
        <div className="legend-item">
          <div className="legend-dot" style={{ background: '#00f0ff' }}></div>
          <span>Active Task</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ background: '#f59e0b' }}></div>
          <span>Critical Path</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ background: '#10b981' }}></div>
          <span>Completed</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ background: '#3b82f6' }}></div>
          <span>Code Module</span>
        </div>
        <div className="legend-item">
          <div className="legend-dot" style={{ background: '#ef4444' }}></div>
          <span>STGAT Fragility α_ij &gt; 0.7</span>
        </div>
      </div>
    </div>
  );
};
