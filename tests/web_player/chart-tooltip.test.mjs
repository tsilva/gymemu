import test from 'node:test';
import assert from 'node:assert/strict';
import { createChartTooltip } from '../../gymemu/web_assets/panels/chart-tooltip.js';

test('tooltip labels the nearest recorded sample without interpolating gaps and clears on hover exit',()=>{
  const previous=globalThis.document;
  const element=()=>({style:{},children:[],offsetWidth:120,offsetHeight:50,
    setAttribute(){},append(...items){this.children.push(...items);},
    replaceChildren(...items){this.children=items;}});
  globalThis.document={createElement:element};
  try {
    const parent=element();
    const canvas={parentElement:parent,offsetLeft:0,offsetTop:0,clientWidth:400,clientHeight:200,addEventListener(){}};
    const render=createChartTooltip(canvas), tooltip=parent.children[0];
    const data={steps:[1,10],series:[{label:'RGB MSE',values:[0.01,0.04],color:'violet'},
      {label:'Missing',values:[null,NaN],color:'gray'}],geometry:{plot:{left:40,right:390,top:10,bottom:160}}};
    render({...data,step:8.5});
    assert.equal(tooltip.hidden,false);
    assert.equal(tooltip.children[0].textContent,'Step 10');
    assert.equal(tooltip.children[1].children[1].textContent,'RGB MSE');
    assert.equal(tooltip.children[1].children[2].textContent,'0.04');
    assert.equal(tooltip.children[2].children[2].textContent,'Unavailable');
    render({...data,step:null});
    assert.equal(tooltip.hidden,true);
    render({...data,steps:[],step:10});
    assert.equal(tooltip.hidden,true);
  } finally {globalThis.document=previous;}
});
