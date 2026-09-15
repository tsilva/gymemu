import test from 'node:test';
import assert from 'node:assert/strict';
import {framePalette,paintLine,changedPixels} from '../../gymemu/web_assets/panels/pixel-editor.js';

test('palette contains only distinct RGB colors already in the frame',()=>{
  const pixels = new Uint8ClampedArray([0,0,0,255,255,0,0,255,0,0,0,255,20,30,40,255]);
  assert.deepEqual(framePalette(pixels),[[0,0,0],[20,30,40],[255,0,0]]);
});
test('fast pointer strokes paint every crossed pixel, including both endpoints',()=>{
  const image = new Uint8ClampedArray(6*4*4), before = image.slice();
  paintLine(image,6,[0,0],[5,3],[255,0,0]);
  assert.deepEqual(changedPixels(before,image).map(p=>p[0]),[0,7,8,15,16,23]);
  assert.ok(changedPixels(before,image).every(p=>p.slice(1).join(',')==='255,0,0'));
  const painted = image.slice();
  paintLine(image,6,[5,3],[0,0],[255,0,0]);
  assert.deepEqual(changedPixels(painted,image),[]);
});
test('single clicks edit exactly one pixel and painting it back removes the change',()=>{
  const image = new Uint8ClampedArray(3*2*4), before = image.slice();
  paintLine(image,3,[2,1],[2,1],[10,20,30]);
  assert.deepEqual(changedPixels(before,image),[[5,10,20,30]]);
  paintLine(image,3,[2,1],[2,1],[0,0,0]);
  assert.deepEqual(changedPixels(before,image),[]);
});

test('the editor explains why Apply is unavailable without blocking inspection',async()=>{
  const {editUnavailableReason}=await import('../../gymemu/web_assets/panels/pixel-editor.js');
  assert.match(editUnavailableReason({history_editable:true,has_prediction:false}),/Run one prediction/);
  assert.match(editUnavailableReason({has_prediction:true}),/Restart the player/);
  assert.equal(editUnavailableReason({history_editable:true,has_prediction:true,loading:null}),null);
});

test('brush size paints its exact square footprint for odd and even sizes',()=>{
  for (const size of [1,2,3,8,32]) {
    const image=new Uint8ClampedArray(64*64*4), before=image.slice();
    paintLine(image,64,[32,32],[32,32],[200,50,60],size);
    assert.equal(changedPixels(before,image).length,size*size);
  }
});
test('large brushes clip to every image edge without wrapping into adjacent rows',()=>{
  for (const [corner,expected] of [
    [[0,0],[0,1,5,6]], [[4,0],[3,4,8,9]],
    [[0,3],[10,11,15,16]], [[4,3],[13,14,18,19]],
  ]) {
    const image=new Uint8ClampedArray(5*4*4), before=image.slice();
    paintLine(image,5,corner,corner,[255,0,0],3);
    assert.deepEqual(changedPixels(before,image).map(p=>p[0]),expected);
  }
});
test('thick strokes stay connected and keep the selected palette color',()=>{
  const image=new Uint8ClampedArray(12*7*4), before=image.slice();
  paintLine(image,12,[2,3],[9,3],[20,30,40],3);
  const changes=changedPixels(before,image);
  assert.equal(changes.length,30);
  assert.ok(changes.every(([index,...rgb])=>index%12>=1&&index%12<=10&&rgb.join(',')==='20,30,40'));
});
