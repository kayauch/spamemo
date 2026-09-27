/** Load ordered photo/support pairs with a bounded number of active requests.
 * On failure, stop scheduling, settle in-flight loads and dispose partial assets.
 * The caller owns every returned texture after successful completion.
 */
export async function loadSurfaceTextures(surfaces, load, {concurrency=8,onProgress}={}) {
  if(!Number.isInteger(concurrency)||concurrency<1)throw new RangeError('Texture concurrency must be a positive integer');
  const paths=surfaces.flatMap(surface=>[surface.texture,surface.support]);
  const loaded=new Array(paths.length);
  let next=0,completed=0,failure=null;
  async function worker(){
    while(failure===null&&next<paths.length){
      const index=next++;
      try{
        loaded[index]=await load(paths[index]);
        completed++;
        onProgress?.(completed,paths.length);
      }catch(error){
        // Preserve the first error even if another in-flight request also fails.
        if(failure===null)failure={error};
      }
    }
  }
  await Promise.all(Array.from({length:Math.min(concurrency,paths.length)},worker));
  if(failure!==null){
    for(const texture of new Set(loaded.filter(Boolean)))texture.dispose?.();
    throw failure.error;
  }
  return surfaces.map((_,index)=>({texture:loaded[index*2],mask:loaded[index*2+1]}));
}
