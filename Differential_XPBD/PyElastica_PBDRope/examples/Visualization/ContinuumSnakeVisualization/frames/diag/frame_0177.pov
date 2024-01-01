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
    ,<0.027207057167420137,-0.0013177628933113551,4.412744525125222>,0.05
    ,<-0.010147881712224232,-0.0011708117811926081,4.445979118167731>,0.05
    ,<-0.0496493317720896,-0.0010371218603615656,4.476637111909724>,0.05
    ,<-0.09129197708599318,-0.0009256184299222659,4.504323336757157>,0.05
    ,<-0.13398587504602907,-0.0008371187630661985,4.530366083441736>,0.05
    ,<-0.17624963270758748,-0.0007610196304453827,4.557107009488602>,0.05
    ,<-0.21624106540001464,-0.0006793246195016536,4.587142631299796>,0.05
    ,<-0.2513228663998786,-0.0005728002585877903,4.622788729505254>,0.05
    ,<-0.2774791574696574,-0.00042935136569681264,4.665413141542898>,0.05
    ,<-0.2894781109783497,-0.00025526908044647606,4.71395390461024>,0.05
    ,<-0.28318426020797993,-8.455120378784491e-05,4.763547803482292>,0.05
    ,<-0.2589359268666617,3.81206985155463e-05,4.807257848591878>,0.05
    ,<-0.2220222628435335,8.522473777015893e-05,4.840960878666562>,0.05
    ,<-0.17908012468525483,5.4088606842054385e-05,4.866550343359704>,0.05
    ,<-0.13517388803746416,-3.8906103439124734e-05,4.8904546782644145>,0.05
    ,<-0.09458556593768983,-0.00016957934678842154,4.919642286547275>,0.05
    ,<-0.062283481633382126,-0.0003153745006532387,4.957798236716388>,0.05
    ,<-0.04243328369883023,-0.00046176668939954533,5.003680602390511>,0.05
    ,<-0.03548685608203344,-0.0006047724730973966,5.0531877652173405>,0.05
    ,<-0.03789880507497994,-0.000747877054511569,5.103122513370279>,0.05
    ,<-0.044509264656249346,-0.0008927011208695729,5.152679352312935>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
