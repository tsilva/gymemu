import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultWorkspace,normalizeWorkspace,saveWorkspace,loadWorkspace,compareWorkspaceRevisions} from '../../gymemu/web_assets/panels/workspace.js';
import {panelDefinition} from '../../gymemu/web_assets/panels/catalog.js';

test('workspace preserves edits, disabled panels, and custom metric widgets',()=>{
  const workspace=defaultWorkspace();
  workspace.panels.prediction.title='Generated';
  workspace.panels.original.enabled=false;
  workspace.panels.history.placement.visible=false;
  workspace.panels.difference.config.gain=4;
  workspace.panels['panel-custom']={type:'telemetry',title:'Latency',enabled:true,config:{view:'stats',metrics:['inference_ms']},placement:{x:2,y:40,w:5,h:6,visible:true}};
  const restored=normalizeWorkspace(workspace);
  assert.equal(restored.panels.prediction.title,'Generated');
  assert.equal(panelDefinition(restored,'original').enabled,false);
  assert.equal(restored.panels.history.placement.visible,false);
  assert.equal(restored.panels.difference.config.gain,4);
  assert.deepEqual(restored.panels['panel-custom'].config.metrics,['inference_ms']);
  assert.equal(restored.panels['panel-custom'].builtin,false);
});

test('workspace rejects executable paths, invalid kinds and unsafe geometry',()=>{
  const workspace=defaultWorkspace();
  workspace.panels.prediction.module='https://evil.example/script.js';
  workspace.panels.prediction.placement={x:100,y:-3,w:100,h:0};
  workspace.panels['panel-evil']={type:'script',module:'evil.js'};
  const restored=normalizeWorkspace(workspace);
  assert.equal(panelDefinition(restored,'prediction').module,'./frame.js');
  assert.equal(restored.panels['panel-evil'],undefined);
  assert.deepEqual(restored.panels.prediction.placement,{x:0,y:0,w:12,h:6,visible:true,window:'main'});
  assert.deepEqual(normalizeWorkspace({version:0}),defaultWorkspace());
});

test('storage failure leaves a usable default workspace',()=>{
  const storage={getItem(){throw Error('disabled')},setItem(){throw Error('disabled')}};
  assert.deepEqual(loadWorkspace(storage),defaultWorkspace());
  assert.doesNotThrow(()=>saveWorkspace(defaultWorkspace(),storage));
});

test('saved layouts drop the retired playback widget and preserve other widgets',()=>{
  for (const version of [1,2]) {
    const saved=defaultWorkspace();
    saved.version=version;
    saved.panels.controls={type:'controls',title:'Playback controls',config:{},placement:{x:8,y:14,w:4,h:10,visible:true}};
    saved.panels.model={type:'telemetry',title:'Model context',config:{view:'stats'},placement:{x:8,y:24,w:4,h:7,visible:false,window:'main'}};
    saved.panels['panel-custom']={...saved.panels.model,title:'Custom stats'};
    saved.panels.prediction.title='My prediction';
    const restored=normalizeWorkspace(saved);
    assert.equal(restored.panels.controls,undefined);
    assert.equal(panelDefinition(restored,'controls'),null);
    assert.equal(restored.panels.model,undefined);
    assert.deepEqual(restored.panels['panel-custom'].placement,{...saved.panels.model.placement,y:10,window:'stats'});
    assert.equal(restored.panels.prediction.title,'My prediction');
    assert.deepEqual(normalizeWorkspace(restored),restored);
  }
});

test('paired workspace puts frames in the player and diagnostic widgets in stats',()=>{
  const workspace=defaultWorkspace();
  assert.deepEqual(Object.entries(workspace.panels).filter(([,p])=>p.placement.window==='main').map(([id])=>id),['original','prediction','difference']);
  assert.deepEqual(Object.entries(workspace.panels).filter(([,p])=>p.placement.window==='stats').map(([id])=>id),['history','metrics']);
  workspace.panels.original.placement.window='stats';
  workspace.panels['panel-custom']={type:'telemetry',title:'Latency',config:{metrics:['inference_ms']},placement:{x:0,y:3,w:4,h:6,window:'main'}};
  const restored=normalizeWorkspace(workspace);
  assert.equal(restored.panels.original.placement.window,'main');
  assert.equal(restored.panels['panel-custom'].placement.window,'stats');
  assert.equal(restored.panels['panel-custom'].placement.y,3);
});

test('legacy diagnostics retain relative positions and preferences in the stats tab',()=>{
  const saved=defaultWorkspace();
  saved.version=2;
  saved.panels.history.placement.y=14;
  saved.panels.metrics.placement.y=24;
  saved.panels.metrics.enabled=false;
  saved.panels.metrics.config.metrics=['mse'];
  const restored=normalizeWorkspace(saved);
  assert.equal(restored.panels.history.placement.y,0);
  assert.equal(restored.panels.metrics.placement.y,10);
  assert.equal(restored.panels.metrics.enabled,false);
  assert.deepEqual(restored.panels.metrics.config.metrics,['mse']);
  assert.deepEqual(normalizeWorkspace(restored),restored);
});

test('workspace revisions order delayed and simultaneous tab updates deterministically',()=>{
  assert.ok(compareWorkspaceRevisions({clock:2,writer:'a'},{clock:1,writer:'z'})>0);
  assert.ok(compareWorkspaceRevisions({clock:2,writer:'b'},{clock:2,writer:'a'})>0);
  assert.equal(compareWorkspaceRevisions({clock:2,writer:'a'},{clock:2,writer:'a'}),0);
  const saved=defaultWorkspace();saved.revision={clock:123,writer:'stats'};
  assert.deepEqual(normalizeWorkspace(saved).revision,saved.revision);
});

test('legacy default frame positions migrate while custom positions survive',()=>{
  const legacy=defaultWorkspace();
  legacy.version=1;
  legacy.panels.prediction.placement.x=0;
  legacy.panels.original.placement.x=4;
  const restored=normalizeWorkspace(legacy);
  assert.equal(restored.panels.original.placement.x,0);
  assert.equal(restored.panels.prediction.placement.x,4);
  assert.equal(restored.panels.difference.placement.x,8);
  assert.deepEqual(normalizeWorkspace(restored),restored);
  legacy.panels.prediction.placement.y=14;
  const custom=normalizeWorkspace(legacy);
  assert.equal(custom.panels.prediction.placement.x,0);
  assert.equal(custom.panels.prediction.placement.y,14);
  assert.equal(custom.panels.original.placement.x,4);
});

test('old default diagnostics migrate to full-width history and an aligned lower row',()=>{
  const saved=defaultWorkspace();
  Object.assign(saved.panels.history.placement,{x:0,y:0,w:8,h:10});
  Object.assign(saved.panels.metrics.placement,{x:0,y:10,w:8,h:7});
  saved.panels.model={type:'telemetry',title:'Model context',placement:{x:8,y:0,w:4,h:7}};
  saved.panels.history.enabled=false;
  saved.panels.difference.config.gain=8;
  const restored=normalizeWorkspace(saved),defaults=defaultWorkspace();
  for(const id of ['history','metrics'])
    assert.deepEqual(restored.panels[id].placement,defaults.panels[id].placement);
  assert.equal(restored.panels.history.enabled,false);
  assert.equal(restored.panels.difference.config.gain,8);
  assert.deepEqual(normalizeWorkspace(restored),restored);
  saved.panels.history.placement.h=12;
  assert.equal(normalizeWorkspace(saved).panels.history.placement.h,12);
});


test('Model context is removed from saved layouts and the former default chart fills its row',()=>{
  const saved=defaultWorkspace();
  saved.panels.metrics.placement.w=8;
  saved.panels.model={type:'telemetry',title:'Model context',placement:{x:8,y:16,w:4,h:11}};
  const restored=normalizeWorkspace(saved);
  assert.equal(restored.panels.model,undefined);
  assert.equal(restored.panels.metrics.placement.w,12);
  assert.equal(panelDefinition(restored,'model'),null);
  assert.deepEqual(normalizeWorkspace(restored),restored);
  saved.panels.metrics.placement.y=25;
  assert.equal(normalizeWorkspace(saved).panels.metrics.placement.w,8);
});
