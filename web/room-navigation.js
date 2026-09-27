// Grounded circular player in a concave room. Geometry is in room Z-up coordinates.
export class RoomNavigation {
  constructor(data) {
    Object.assign(this, data);
    this.radius = data.radius ?? .18;
    this.eye_height = data.eye_height ?? 1.55;
    this.colliders = data.colliders || [];
    if (!Array.isArray(this.footprint) || this.footprint.length < 3 ||
        !this.footprint.flat().every(Number.isFinite) || !(this.ceiling > this.floor)) throw Error('歩行範囲が不正です');
  }
  valid(x,y) {
    if (!Number.isFinite(x+y)) return false;
    let inside=false;
    for(let i=0,j=this.footprint.length-1;i<this.footprint.length;j=i++) {
      const a=this.footprint[j], b=this.footprint[i], dx=b[0]-a[0],dy=b[1]-a[1];
      if((a[1]>y)!==(b[1]>y) && x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]) inside=!inside;
      const t=Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy || 1)));
      if(Math.hypot(x-a[0]-t*dx,y-a[1]-t*dy)<this.radius) return false;
    }
    if(!inside) return false;
    for(const b of this.colliders) {
      if(b.top<=this.floor+.03 || b.bottom>=this.floor+this.eye_height+.2) continue;
      const dx=x-b.center[0],dy=y-b.center[1],r=b.rotation;
      const localX=dx*r[0][0]+dy*r[1][0],localY=dx*r[0][1]+dy*r[1][1];
      const ex=Math.max(0,Math.abs(localX)-b.scale[0]/2),ey=Math.max(0,Math.abs(localY)-b.scale[1]/2);
      if(Math.hypot(ex,ey)<this.radius) return false;
    }
    return true;
  }
  findSpawn(preferred) {
    if(preferred && this.valid(...preferred)) return {x:preferred[0],y:preferred[1],z:this.floor+this.eye_height};
    const xs=this.footprint.map(p=>p[0]),ys=this.footprint.map(p=>p[1]);
    const lo=[Math.min(...xs),Math.min(...ys)],hi=[Math.max(...xs),Math.max(...ys)];
    const target=preferred || [(lo[0]+hi[0])/2,(lo[1]+hi[1])/2];
    let result=null,best=Infinity;
    const step=Math.max(.06,Math.max(hi[0]-lo[0],hi[1]-lo[1])/120);
    for(let x=lo[0]+this.radius;x<hi[0];x+=step)for(let y=lo[1]+this.radius;y<hi[1];y+=step){
      const distance=Math.hypot(x-target[0],y-target[1]);
      if(distance<best&&this.valid(x,y)){best=distance;result={x,y,z:this.floor+this.eye_height};}
    }
    return result;
  }
  move(position, motion) {
    let x=position.x,y=position.y;
    if(!this.valid(x,y)) return this.findSpawn([x,y]) || {...position};
    const steps=Math.max(1,Math.ceil(Math.hypot(motion.x,motion.y)/(this.radius*.4)));
    if(steps>10000) return {x,y,z:this.floor+this.eye_height};
    const dx=motion.x/steps,dy=motion.y/steps;
    for(let i=0;i<steps;i++){
      if(this.valid(x+dx,y+dy)){x+=dx;y+=dy;}
      else if(Math.abs(dx)>Math.abs(dy)){
        if(this.valid(x+dx,y))x+=dx;
        if(this.valid(x,y+dy))y+=dy;
      }else{
        if(this.valid(x,y+dy))y+=dy;
        if(this.valid(x+dx,y))x+=dx;
      }
    }
    return {x,y,z:this.floor+this.eye_height};
  }
}
