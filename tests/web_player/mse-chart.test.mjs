import test from 'node:test';
import assert from 'node:assert/strict';
import {measuredHistory,chartPoint} from '../../gymemu/web_assets/panels/telemetry.js';
import {selectedRange,resizeChartRange,bindChartRange} from '../../gymemu/web_assets/chart-range.js';

const snapshot={mode:'teacher-forcing',history:[{step:1,mse:.1},{step:2,mse:.8},{step:100,mse:.3},{step:101,mse:null}]};
test('MSE chart filters invalid and unscored points, preserves timestep gaps',()=>{
  assert.deepEqual(measuredHistory(snapshot).map(p=>p.step),[1,2,100]);
  assert.deepEqual(measuredHistory(snapshot,{first:2,last:50}),[{step:2,mse:.8}]);
  assert.deepEqual(measuredHistory({...snapshot,mode:'autoregressive'}),[]);
});
test('chart selection uses actual timestep coordinates instead of treating steps as array offsets',()=>{
  const history=measuredHistory(snapshot),geometry={plot:{left:10,right:1000,positions:[10,20,1000]}};
  assert.equal(chartPoint(history,geometry,24).step,2);
  assert.equal(chartPoint(history,geometry,900).step,100);
  assert.equal(chartPoint(history,geometry,1001),null);
  assert.equal(chartPoint([],geometry,100),null);
});
test('drag zoom clamps to plot, supports reverse drags and ignores clicks',()=>{
  const plot={left:10,right:110};
  assert.deepEqual(selectedRange(plot,90,30,1,101),{first:21,last:81});
  assert.deepEqual(selectedRange(plot,-100,200,1,101),{first:1,last:101});
  assert.equal(selectedRange(plot,30,33,1,101),null);
  assert.equal(selectedRange(plot,30,90,1,1),null);
});
test('timeline zoom handles cannot cross or leave the episode',()=>{
  const range={first:10,last:40},episode={first:0,last:100};
  assert.deepEqual(resizeChartRange(range,episode,'first',50),{first:39,last:40});
  assert.deepEqual(resizeChartRange(range,episode,'last',5),{first:10,last:11});
  assert.deepEqual(resizeChartRange(range,episode,'first',-20),{first:0,last:40});
  assert.deepEqual(resizeChartRange(range,episode,'last',200),{first:10,last:100});
});

test('zoom drag suppresses seeking and double-click resets the range',()=>{
  const listeners=new Map(),ranges=[];
  const oldDocument=globalThis.document;
  globalThis.document={createElement:()=>({style:{},hidden:false})};
  const canvas={style:{},clientWidth:200,offsetLeft:0,offsetTop:0,
    parentElement:{style:{},append(){}},
    getBoundingClientRect:()=>({left:0,width:200}),
    setPointerCapture(){},releasePointerCapture(){},
    addEventListener(name,handler){listeners.set(name,handler);}};
  try {
    bindChartRange(canvas,()=>({plot:{left:0,right:200,top:0,bottom:100}}),
      ()=>({history:[{step:1},{step:101}]}),{setChartRange:range=>ranges.push(range)});
    const event=x=>({clientX:x,button:0,pointerId:1,preventDefault(){}});
    listeners.get('pointerdown')(event(20));
    listeners.get('pointermove')(event(140));
    listeners.get('pointerup')(event(140));
    assert.deepEqual(ranges,[{first:11,last:71}]);
    let suppressed=false;
    listeners.get('click')({stopImmediatePropagation(){suppressed=true;}});
    assert.equal(suppressed,true);
    listeners.get('dblclick')(event(80));
    assert.equal(ranges.at(-1),null);
    suppressed=false;
    listeners.get('pointerdown')(event(40));
    listeners.get('pointerup')(event(40));
    listeners.get('click')({stopImmediatePropagation(){suppressed=true;}});
    assert.equal(suppressed,false);
    assert.equal(ranges.length,2);
  } finally {globalThis.document=oldDocument;}
});
