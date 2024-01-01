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
    ,<-0.055539761615998454,-7.640256006025743e-05,0.03012838379727875>,0.05
    ,<-0.031620007155400476,-9.879628843215149e-05,0.07403610641157043>,0.05
    ,<-0.008385895364905728,-0.00011918983136886148,0.11831439759723368>,0.05
    ,<0.012732684126986964,-0.0001320375729105097,0.1636432335217829>,0.05
    ,<0.03017284581723075,-0.00013076614485546875,0.21051215638505172>,0.05
    ,<0.04257320909447385,-0.00011054739927939539,0.2589587200184773>,0.05
    ,<0.04888319835384357,-7.116861908504569e-05,0.3085658990976864>,0.05
    ,<0.04834203266796445,-2.0252258201596173e-05,0.3585667024054142>,0.05
    ,<0.04060371523443318,2.8337558872533853e-05,0.40796573757815474>,0.05
    ,<0.026178990540083134,6.098782803908021e-05,0.4558409697119144>,0.05
    ,<0.00680265548293927,6.99354314822048e-05,0.5019371184999549>,0.05
    ,<-0.014673262937019068,5.626882622825999e-05,0.5470970975095678>,0.05
    ,<-0.03464725719596673,2.914686718577758e-05,0.5929447159255908>,0.05
    ,<-0.049220099711800046,1.6431718090691496e-06,0.6407856659977854>,0.05
    ,<-0.05502020820920097,-1.5159922937546895e-05,0.6904590786716254>,0.05
    ,<-0.05047407171186176,-1.7434152247655434e-05,0.7402580620297102>,0.05
    ,<-0.036448746988681106,-1.0398283495176019e-05,0.788248809484884>,0.05
    ,<-0.015584889976275032,-4.117544171214228e-06,0.8336792120883424>,0.05
    ,<0.008967093862971256,-4.424707944715595e-06,0.8772265272985589>,0.05
    ,<0.03459133965314905,-1.0100304313874614e-05,0.9201554762217207>,0.05
    ,<0.05990717655029572,-1.7682597706825095e-05,0.9632704691455255>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
