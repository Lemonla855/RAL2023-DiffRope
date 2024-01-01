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
    ,<0.00023178015828226534,1.5246569471406246e-08,5.959607409049798e-06>,0.05
    ,<9.78880682528333e-05,1.9553759595185406e-08,0.050005776838237016>,0.05
    ,<-7.213633102018702e-06,1.681499893767085e-08,0.10000563568481391>,0.05
    ,<-6.840937600095508e-05,5.268735300913747e-09,0.15000563303527598>,0.05
    ,<-9.707697482083445e-05,-3.8618174141100765e-09,0.20000557448743286>,0.05
    ,<-0.00010801531320906472,-9.863059541349762e-09,0.2500056136523936>,0.05
    ,<-0.00011804123178627572,-2.0544855438802162e-08,0.3000056417848137>,0.05
    ,<-0.00014057837254756968,-3.924248874015276e-08,0.3500056254597599>,0.05
    ,<-0.00016228754705859458,-5.1843407144445604e-08,0.4000056793592758>,0.05
    ,<-0.00014546177085603248,-2.9099314480153238e-08,0.45000573503033625>,0.05
    ,<-5.789776214768827e-05,3.7657688727974637e-08,0.5000056991364266>,0.05
    ,<0.00012805531797703277,7.861968723786228e-08,0.5500054391189891>,0.05
    ,<0.00038229757579693745,4.372001085488295e-08,0.6000049438977183>,0.05
    ,<0.0005955255775421446,-3.028964820966975e-08,0.6500050382701831>,0.05
    ,<0.0006612101412668741,-5.906719599717245e-08,0.7000057760956618>,0.05
    ,<0.0005431522572085466,-8.386807326445901e-09,0.7500062795685343>,0.05
    ,<0.0002921119865947926,6.73644343219836e-08,0.8000061414228643>,0.05
    ,<6.433890913663782e-06,8.670817783620562e-08,0.8500055775200405>,0.05
    ,<-0.00024665325917228514,2.5868835638011992e-08,0.9000050798229837>,0.05
    ,<-0.00043259978187759423,-6.282553991829675e-08,0.9500049092912797>,0.05
    ,<-0.0005441565059964034,-1.4642529915160713e-07,1.00000499809642>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
