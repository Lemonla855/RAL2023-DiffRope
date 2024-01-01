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
    ,<0.021827651299109713,-0.002386827391793654,3.177076694155115>,0.05
    ,<0.03060877831645521,-0.0019237955703395354,3.2262925500382815>,0.05
    ,<0.03359223289783084,-0.0014609103397681153,3.2762000502124713>,0.05
    ,<0.026877352014510286,-0.001010354511959462,3.3257463785918246>,0.05
    ,<0.009114527398832091,-0.0006016225618865993,3.3724858507092796>,0.05
    ,<-0.018704744496787554,-0.0002748960831177088,3.4140355049401596>,0.05
    ,<-0.054180999506443395,-6.160286206695405e-05,3.449278307292864>,0.05
    ,<-0.09449922864745941,2.987178838332371e-05,3.478866541110209>,0.05
    ,<-0.13692199975359698,1.4498533397254036e-05,3.5053559489184187>,0.05
    ,<-0.178543288916919,-7.446766725788506e-05,3.5330917002675166>,0.05
    ,<-0.21531158720998164,-0.00019374363566844872,3.566997914744819>,0.05
    ,<-0.24108753915879527,-0.00030041986556728024,3.6098543600819766>,0.05
    ,<-0.24913010327134827,-0.0003650659443162308,3.659205067612231>,0.05
    ,<-0.236637791364578,-0.00038098690204583336,3.7076109724859956>,0.05
    ,<-0.20746134791740753,-0.000362837910499793,3.7481976460403126>,0.05
    ,<-0.1689207724330289,-0.00033448264692497354,3.7800256996027652>,0.05
    ,<-0.12718095804683396,-0.00031139528079304014,3.8075283518612832>,0.05
    ,<-0.08637088337237447,-0.00029418036002036173,3.836397920848214>,0.05
    ,<-0.0495037818943202,-0.00027546825574430205,3.8701621292081376>,0.05
    ,<-0.018015734678908917,-0.0002489817968008177,3.9089948318409555>,0.05
    ,<0.009478086144689922,-0.00021545116115678396,3.9507536402538093>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
