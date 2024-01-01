#include "../default.inc"

camera{
    location <15.0,10.5,-15.0>
    angle 30
    look_at <4.0,2.7,2.0>
    sky <0,1,0>
    right x*image_width/image_height
}
light_source{
    <15.0,10.5,-15.0>
    color rgb<0.09,0.09,0.1>
}
light_source{
    <1500,2500,-1000>
    color White
}

sphere_sweep {
    linear_spline 21
    ,<-0.04005275257440856,-0.0008152144889573805,3.9398900427885835>,0.05
    ,<-0.08597598802892588,-0.0009090246806751475,3.9596701947520105>,0.05
    ,<-0.1311913462189072,-0.000990898437885048,3.9810268813622103>,0.05
    ,<-0.17396995462138126,-0.0010384189843961097,4.006921981916621>,0.05
    ,<-0.21154011956571137,-0.0010297754072439247,4.0399163424601126>,0.05
    ,<-0.24024017219578747,-0.0009566402847185436,4.080849795802501>,0.05
    ,<-0.2562544937476717,-0.0008297095283547081,4.128197476771083>,0.05
    ,<-0.2567142951964322,-0.0006732664628538848,4.178171139735358>,0.05
    ,<-0.2408139060566179,-0.0005183745643041028,4.22554908479141>,0.05
    ,<-0.21077607685216035,-0.0003950690279227969,4.265496982690845>,0.05
    ,<-0.1714218371870993,-0.00032187467679776826,4.29632609312693>,0.05
    ,<-0.1279356544956465,-0.00029950080676424684,4.321006451330353>,0.05
    ,<-0.08461606263234078,-0.00031397749208074,4.34598247873182>,0.05
    ,<-0.04624282234973468,-0.0003424574816744944,4.378033753066561>,0.05
    ,<-0.01948881970652168,-0.00036212143628625547,4.420258298480919>,0.05
    ,<-0.010425245278029419,-0.00036243808496662496,4.469406867677066>,0.05
    ,<-0.019950796173771452,-0.00035034852935817123,4.518463984114752>,0.05
    ,<-0.04344782390512322,-0.0003449166463293466,4.562571113703116>,0.05
    ,<-0.07458777696232263,-0.00035842332039929213,4.601667887491275>,0.05
    ,<-0.10841055865708882,-0.00038786030533801194,4.638478115013806>,0.05
    ,<-0.14224741094352383,-0.0004244733559481834,4.675282057793789>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
