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
    ,<-0.28165507116810784,-0.0023254915448520613,4.029955614052577>,0.05
    ,<-0.2746240114894497,-0.0018753349324869284,4.079437746435907>,0.05
    ,<-0.26216691826584815,-0.001434677549394594,4.12784424290338>,0.05
    ,<-0.24179297288251553,-0.0010227500812085154,4.173493339221236>,0.05
    ,<-0.21365244702962904,-0.0006694841042190265,4.214817379438439>,0.05
    ,<-0.17949742661904933,-0.00040133066503643703,4.251338125997044>,0.05
    ,<-0.14161531204026354,-0.00022674918056417995,4.28398807963026>,0.05
    ,<-0.10233289258022212,-0.00013471279015758557,4.314947101631115>,0.05
    ,<-0.06432264747213294,-9.853344165209069e-05,4.347457118478088>,0.05
    ,<-0.03148131811495293,-8.360513048685615e-05,4.385175641781385>,0.05
    ,<-0.00951777567117033,-5.9052525016778116e-05,4.4300960255256845>,0.05
    ,<-0.004659928600610331,-1.2619181955087833e-05,4.4798481465431275>,0.05
    ,<-0.01978535693887238,3.710889570840468e-05,4.527480322579862>,0.05
    ,<-0.05154157807292308,5.80358157848219e-05,4.566065955479284>,0.05
    ,<-0.09281118849234536,3.164450198076418e-05,4.59425569788507>,0.05
    ,<-0.13746908118987525,-4.1871871833226326e-05,4.616710791305721>,0.05
    ,<-0.18157732416319897,-0.000148113751037077,4.6402348002602>,0.05
    ,<-0.2218104051823674,-0.0002671242806734725,4.669903328932286>,0.05
    ,<-0.25529877235555026,-0.0003821744151046174,4.707016089267532>,0.05
    ,<-0.28146284596445315,-0.0004859212116341985,4.749610775410575>,0.05
    ,<-0.30298428708311226,-0.0005810404456018466,4.7947316310765915>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }