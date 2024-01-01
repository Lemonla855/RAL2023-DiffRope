#include "../default.inc"

camera{
    location <0,15,3>
    angle 30
    look_at <0.0,0,3>
    sky <-1,0,0>
    right x*image_width/image_height
}
light_source{
    <0.0,8.0,5.0>
    color rgb<0.09,0.09,0.1>
}
light_source{
    <1500,2500,-1000>
    color White
}

sphere_sweep {
    linear_spline 21
    ,<-0.19247818854787244,-0.0005538943577542492,2.551054074132793>,0.05
    ,<-0.14918698284531842,-0.0006632257698235033,2.5760715192508017>,0.05
    ,<-0.10563531482592049,-0.0007733882767219875,2.600642775487978>,0.05
    ,<-0.06281484943195237,-0.0008724897826587363,2.626475589526434>,0.05
    ,<-0.022533711506026708,-0.0009390718447521615,2.656114399924927>,0.05
    ,<0.012635371362766146,-0.0009490930949691461,2.6916646586580075>,0.05
    ,<0.0393799443692581,-0.0008851535960203721,2.7339094887968867>,0.05
    ,<0.05390581190411453,-0.0007475358390704409,2.7817396897140982>,0.05
    ,<0.05267992774488856,-0.000561860639581708,2.8317006314119215>,0.05
    ,<0.03428063678458832,-0.00037088531434287334,2.878161003710252>,0.05
    ,<0.0012896062474798118,-0.00021659280694110645,2.915699083464736>,0.05
    ,<-0.04070903539290016,-0.00012545285057702002,2.9428038352054298>,0.05
    ,<-0.08630898806208606,-0.00010160099706665525,2.963301879565747>,0.05
    ,<-0.13162288717400702,-0.00013144457207397536,2.9844296709688036>,0.05
    ,<-0.17242236303886563,-0.000193490197562885,3.0133166721112126>,0.05
    ,<-0.202717665762254,-0.00026780795604837096,3.05306813238197>,0.05
    ,<-0.21731838614392152,-0.0003434734506554207,3.1008594336157045>,0.05
    ,<-0.21570484778660237,-0.000421526738023546,3.150803533457419>,0.05
    ,<-0.20207305187222152,-0.0005095790019891198,3.198882730997832>,0.05
    ,<-0.1821356097593548,-0.000609350748915853,3.244715842875061>,0.05
    ,<-0.16037542184338363,-0.0007145090978591737,3.2897189052173323>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
