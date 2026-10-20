import React, { useState } from 'react';

interface InterventionModalProps {
  isOpen: boolean;
  onClose: () => void;
  tasks: Record<string, any>;
  onApplied: () => void;
}

export const InterventionModal: React.FC<InterventionModalProps> = ({
  isOpen,
  onClose,
  tasks,
  onApplied,
}) => {
  const [interventionType, setInterventionType] = useState('PRIORITY_OVERRIDE');
  const [selectedTask, setSelectedTask] = useState(Object.keys(tasks)[0] || 'TASK-01');
  const [priority, setPriority] = useState(5);
  const [bufferHours, setBufferHours] = useState(2.0);
  const [newTitle, setNewTitle] = useState('Manual Security Review');
  const [newDuration, setNewDuration] = useState(4.0);
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    let params: Record<string, any> = {};
    if (interventionType === 'PRIORITY_OVERRIDE') {
      params = { task_id: selectedTask, priority };
    } else if (interventionType === 'INJECT_BUFFER') {
      params = { task_id: selectedTask, buffer_hours: bufferHours };
    } else if (interventionType === 'INSERT_TASK') {
      params = {
        task_id: `MANUAL-${Math.floor(Math.random() * 900 + 100)}`,
        title: newTitle,
        duration: newDuration,
        priority,
      };
    }

    try {
      await fetch('/api/orchestrator/intervention', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ intervention_type: interventionType, params }),
      });
      onApplied();
      onClose();
    } catch {
      onApplied();
      onClose();
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="modal-overlay" id="intervention-modal-overlay">
      <div className="modal-content glass-panel">
        <div className="modal-header">
          <div className="modal-title">Human-in-the-Loop Intervention</div>
          <button
            onClick={onClose}
            style={{ background: 'transparent', border: 'none', color: '#94a3b8', fontSize: '1.25rem', cursor: 'pointer' }}
          >
            &times;
          </button>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="form-group">
            <label>Intervention Action</label>
            <select
              id="intervention-type-select"
              value={interventionType}
              onChange={(e) => setInterventionType(e.target.value)}
            >
              <option value="PRIORITY_OVERRIDE">Override Task Priority (1 - 5)</option>
              <option value="INJECT_BUFFER">Inject Safety Buffer Margin (+hours)</option>
              <option value="INSERT_TASK">Inject Ad-hoc Work Item</option>
            </select>
          </div>

          {interventionType !== 'INSERT_TASK' && (
            <div className="form-group">
              <label>Target Task</label>
              <select
                id="intervention-task-select"
                value={selectedTask}
                onChange={(e) => setSelectedTask(e.target.value)}
              >
                {Object.keys(tasks).map((tid) => (
                  <option key={tid} value={tid}>
                    {tid} — {tasks[tid]?.title || ''}
                  </option>
                ))}
              </select>
            </div>
          )}

          {interventionType === 'PRIORITY_OVERRIDE' && (
            <div className="form-group">
              <label>New Priority (1: Low to 5: Critical)</label>
              <input
                id="intervention-priority-input"
                type="number"
                min="1"
                max="5"
                value={priority}
                onChange={(e) => setPriority(parseInt(e.target.value) || 1)}
              />
            </div>
          )}

          {interventionType === 'INJECT_BUFFER' && (
            <div className="form-group">
              <label>Safety Buffer Hours to Add</label>
              <input
                id="intervention-buffer-input"
                type="number"
                step="0.5"
                min="0.5"
                max="24"
                value={bufferHours}
                onChange={(e) => setBufferHours(parseFloat(e.target.value) || 1)}
              />
            </div>
          )}

          {interventionType === 'INSERT_TASK' && (
            <>
              <div className="form-group">
                <label>Task Title</label>
                <input
                  id="intervention-new-title-input"
                  type="text"
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label>Estimated Duration (hours)</label>
                <input
                  id="intervention-new-duration-input"
                  type="number"
                  step="0.5"
                  value={newDuration}
                  onChange={(e) => setNewDuration(parseFloat(e.target.value) || 1)}
                />
              </div>
            </>
          )}

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '0.5rem' }}>
            <button type="button" className="btn btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              id="intervention-apply-btn"
              type="submit"
              className="btn btn-primary"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Applying...' : 'Apply Intervention'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
