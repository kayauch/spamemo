// 2D oriented disks, kept in the original VGGT coordinate system.
export class SplatLayer {
  constructor(scene, loadLibrary = () => import('/static/vendor/gaussian-splats-3d.module.js')) {
    this.scene = scene;
    this.loadLibrary = loadLibrary;
    this.current = null;
    this.generation = 0;
  }

  clear() {
    this.generation++;
    if (this.current) {
      const old = this.current;
      this.current = null;
      this.scene.remove(old);
      this.release(old);
    }
  }

  async release(viewer) {
    try { await viewer.dispose(); }
    catch (error) { console.warn('2DGS disposal:', error); }
    viewer.callbackMesh?.geometry?.dispose();
    viewer.callbackMesh?.material?.dispose();
  }

  setVisible(value) { if (this.current) this.current.visible = value; }

  async load(url) {
    this.clear();
    const generation = this.generation;
    const GS = await this.loadLibrary();
    if (generation !== this.generation) return false;
    const viewer = new GS.DropInViewer({
      splatRenderMode: GS.SplatRenderMode.TwoD,
      sharedMemoryForWorkers: false,
      gpuAcceleratedSort: false,
      integerBasedSort: false,
      sphericalHarmonicsDegree: 2,
      sceneRevealMode: GS.SceneRevealMode.Instant,
    });
    try {
      await viewer.addSplatScene(url, { format: GS.SceneFormat.Ply, showLoadingUI: false,
        splatAlphaRemovalThreshold: 1, position: [0,0,0], rotation: [0,0,0,1], scale: [1,1,1] });
      if (generation !== this.generation) { await this.release(viewer); return false; }
      this.current = viewer;
      this.scene.add(viewer);
      return true;
    } catch (error) {
      await this.release(viewer);
      throw error;
    }
  }
}
