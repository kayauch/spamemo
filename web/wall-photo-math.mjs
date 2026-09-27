export function validateQuad(q){
 if(!Array.isArray(q)||q.length!==4||q.some(p=>p.length!==2||p.some(v=>!Number.isFinite(v)||v<0||v>1)))throw Error('写真の中の四隅を指定してください。');
 const cross=q.map((p,i)=>{const b=q[(i+1)%4],c=q[(i+2)%4];return (b[0]-p[0])*(c[1]-b[1])-(b[1]-p[1])*(c[0]-b[0]);});
 if(cross.some(v=>v<.0001))throw Error('左上→右上→右下→左下の順で、交差しない四隅を指定してください。');
 const area=Math.abs(q.reduce((a,p,i)=>a+p[0]*q[(i+1)%4][1]-p[1]*q[(i+1)%4][0],0))/2;
 if(area<.005)throw Error('選んだ範囲が小さすぎます。');
 return q;
}
export function homography(q){
 validateQuad(q);const dest=[[0,0],[1,0],[1,1],[0,1]],a=[];
 dest.forEach(([u,v],i)=>{const [x,y]=q[i];a.push([u,v,1,0,0,0,-u*x,-v*x,x],[0,0,0,u,v,1,-u*y,-v*y,y]);});
 for(let c=0;c<8;c++){let p=c;for(let r=c+1;r<8;r++)if(Math.abs(a[r][c])>Math.abs(a[p][c]))p=r;
  [a[p],a[c]]=[a[c],a[p]];if(Math.abs(a[c][c])<1e-10)throw Error('四隅をもう一度指定してください。');
  const d=a[c][c];for(let j=c;j<=8;j++)a[c][j]/=d;
  for(let r=0;r<8;r++)if(r!==c){const k=a[r][c];for(let j=c;j<=8;j++)a[r][j]-=k*a[c][j];}
 }return [...a.map(row=>row[8]),1];
}
export function project(h,u,v){const d=h[6]*u+h[7]*v+1;return [(h[0]*u+h[1]*v+h[2])/d,(h[3]*u+h[4]*v+h[5])/d];}
export function surfaceSignature(s){return JSON.stringify([s.id,...['origin','u','v'].map(k=>s[k].map(n=>Math.round(n*10000)/10000))]);}
export function warpPixels(src,width,height,quad,outWidth,outHeight){
 const h=homography(quad),out=new Uint8ClampedArray(outWidth*outHeight*4);
 for(let y=0;y<outHeight;y++)for(let x=0;x<outWidth;x++){
  const [u,v]=project(h,(x+.5)/outWidth,(y+.5)/outHeight),sx=Math.max(0,Math.min(width-1,u*(width-1))),sy=Math.max(0,Math.min(height-1,v*(height-1)));
  const x0=Math.floor(sx),y0=Math.floor(sy),x1=Math.min(x0+1,width-1),y1=Math.min(y0+1,height-1),fx=sx-x0,fy=sy-y0;
  for(let c=0;c<4;c++)out[(y*outWidth+x)*4+c]=(src[(y0*width+x0)*4+c]*(1-fx)+src[(y0*width+x1)*4+c]*fx)*(1-fy)+(src[(y1*width+x0)*4+c]*(1-fx)+src[(y1*width+x1)*4+c]*fx)*fy;
 }return out;
}
