import * as THREE from 'three';

// Camera-relative walking, independent of VGGT's arbitrary world-up direction.
export class WalkControls {
  constructor(camera, canvas, getControls, onExit) {
    this.camera = camera;
    this.canvas = canvas;
    this.getControls = getControls;
    this.onExit = onExit;
    this.active = false;
    this.keys = new Set();
    this.buttons = new Map();
    this.abort = new AbortController();
    const listen = (target, event, fn) => target.addEventListener(event, fn, {signal:this.abort.signal});
    const editing = target => target?.closest?.('input, textarea, select, [contenteditable="true"]');
    listen(window, 'keydown', event => {
      if (!this.active) return;
      if (event.code === 'Escape') { this.onExit(); return; }
      if (editing(event.target) || event.ctrlKey || event.metaKey || event.altKey) return;
      if (['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(event.code)) {
        event.preventDefault(); this.keys.add(event.code);
      }
    });
    listen(window, 'keyup', event => this.keys.delete(event.code));
    listen(window, 'blur', () => this.clear());
    listen(document, 'visibilitychange', () => this.clear());
    listen(document, 'focusin', event => { if (editing(event.target)) this.clear(); });
    listen(canvas, 'pointerdown', event => {
      if (!this.active || event.button > 0) return;
      this.drag = {id:event.pointerId, x:event.clientX, y:event.clientY};
      canvas.setPointerCapture(event.pointerId);
    });
    listen(canvas, 'pointermove', event => {
      if (!this.active || this.drag?.id !== event.pointerId) return;
      const dx = event.clientX-this.drag.x, dy = event.clientY-this.drag.y;
      this.drag.x=event.clientX; this.drag.y=event.clientY;
      const forward = camera.getWorldDirection(new THREE.Vector3());
      forward.applyAxisAngle(this.up, -dx*.003);
      const pitch = Math.asin(THREE.MathUtils.clamp(forward.dot(this.up), -1, 1));
      const right = new THREE.Vector3().crossVectors(forward, this.up).normalize();
      const next = THREE.MathUtils.clamp(pitch-dy*.003, -1.45, 1.45);
      forward.applyAxisAngle(right, next-pitch);
      camera.lookAt(camera.position.clone().add(forward));
      this.syncTarget();
    });
    listen(canvas, 'lostpointercapture', () => { this.drag=null; });
    listen(window, 'pointerup', event => { this.buttons.delete(event.pointerId); if (this.drag?.id===event.pointerId) this.drag=null; });
    listen(window, 'pointercancel', event => { this.buttons.delete(event.pointerId); this.drag=null; });
    for (const button of document.querySelectorAll('[data-walk-key]')) {
      listen(button, 'pointerdown', event => {
        if (!this.active) return;
        event.preventDefault(); button.setPointerCapture(event.pointerId);
        this.buttons.set(event.pointerId, button.dataset.walkKey);
      });
      listen(button, 'lostpointercapture', event => this.buttons.delete(event.pointerId));
    }
  }
  clear() { this.keys.clear(); this.buttons.clear(); this.drag=null; this.lastTime=null; }
  start(diagonal) {
    this.clear(); this.active=true;
    this.up=this.camera.up.clone().normalize();
    this.speed=Math.max(diagonal*.08, .001);
    this.getControls().enabled=false;
    this.canvas.style.cursor='grab';
  }
  stop() {
    this.active=false; this.clear(); this.syncTarget();
    this.getControls().enabled=true;
    this.canvas.style.cursor='';
  }
  syncTarget() {
    this.getControls().target.copy(this.camera.position).add(this.camera.getWorldDirection(new THREE.Vector3()));
  }
  update(now) {
    if (!this.active) return;
    const dt=this.lastTime == null ? 0 : Math.min((now-this.lastTime)/1000, .05);
    this.lastTime=now;
    const held=new Set([...this.keys, ...this.buttons.values()]);
    const axis=(positive, negative) => Number(positive.some(k=>held.has(k)))-Number(negative.some(k=>held.has(k)));
    const forward=this.camera.getWorldDirection(new THREE.Vector3()).projectOnPlane(this.up).normalize();
    const right=new THREE.Vector3().crossVectors(forward, this.up).normalize();
    const motion=forward.multiplyScalar(axis(['KeyW','ArrowUp'], ['KeyS','ArrowDown']))
      .addScaledVector(right, axis(['KeyD','ArrowRight'], ['KeyA','ArrowLeft']))
      .addScaledVector(this.up, axis(['KeyE'], ['KeyQ']));
    if (motion.lengthSq()>0) {
      const delta=motion.normalize().multiplyScalar(this.speed*dt);
      if(this.constrainMove) this.camera.position.copy(this.constrainMove(this.camera.position,delta));
      else this.camera.position.add(delta);
    }
    this.syncTarget();
  }
  dispose() { this.stop(); this.abort.abort(); }
}
