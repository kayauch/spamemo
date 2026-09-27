import test from 'node:test';
import assert from 'node:assert/strict';
import {captureOrientation} from '../web/capture-orientation.mjs';
const landscape = {landscape: true, width: 1920, height: 1080, angle: 90};
test('portrait display cannot start even when the camera supplies wide frames', () => {
  assert.equal(captureOrientation({...landscape, landscape: false}).reason, 'portrait');
});
test('landscape display requires actual landscape video and loaded metadata', () => {
  assert.equal(captureOrientation({...landscape, width: 1080, height: 1920}).reason, 'portrait-frame');
  assert.equal(captureOrientation({...landscape, width: 0, height: 0}).reason, 'loading');
  assert.equal(captureOrientation(landscape).allowed, true);
});
test('turning portrait, flipping landscape, or resizing frames ends a segment', () => {
  for (const changed of [{...landscape, landscape: false}, {...landscape, angle: 270}, {...landscape, width: 1280, height: 720}]) {
    assert.equal(captureOrientation(changed, landscape).allowed, false);
  }
  assert.equal(captureOrientation(landscape, landscape).allowed, true);
});
test('browsers without angle reporting still use viewport and real frame dimensions', () => {
  const noAngle = {...landscape, angle: null};
  assert.equal(captureOrientation(noAngle, noAngle).allowed, true);
  assert.equal(captureOrientation({...noAngle, landscape: false}, noAngle).allowed, false);
});
