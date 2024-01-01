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
    ,<0.056277080085086514,-0.0018487434146349515,3.8196675052369313>,0.05
    ,<0.03063787946941001,-0.0015225188211113539,3.8625662808663717>,0.05
    ,<0.001040222278570467,-0.001206997512389981,3.9028438511453096>,0.05
    ,<-0.033519624893879205,-0.0009186583708726378,3.9389659610234036>,0.05
    ,<-0.07211268801823331,-0.0006779009127703036,3.9707572671795974>,0.05
    ,<-0.11308535538473882,-0.000497438689479093,3.9994318970427343>,0.05
    ,<-0.15466878445331944,-0.0003738377847485813,4.027223956592002>,0.05
    ,<-0.19487538199045895,-0.00028969517340408127,4.056974345569845>,0.05
    ,<-0.23077705324504436,-0.0002195200674995318,4.091791131069455>,0.05
    ,<-0.25767476097522185,-0.0001399314843801728,4.1339395424849625>,0.05
    ,<-0.26959151830590833,-4.4608747361852543e-05,4.182482075697626>,0.05
    ,<-0.2622566338288445,4.6281488659923255e-05,4.231911822651061>,0.05
    ,<-0.23677046426396914,0.00010174491720781349,4.274892733292823>,0.05
    ,<-0.19923498760531388,0.00010284745178246116,4.307889567783489>,0.05
    ,<-0.15643026796675175,4.726579844693864e-05,4.333705513774491>,0.05
    ,<-0.11327119996771215,-5.4149194750499784e-05,4.358934002501388>,0.05
    ,<-0.07386320874693086,-0.00018315940172076571,4.389690772630257>,0.05
    ,<-0.042357880279027725,-0.0003226989788070949,4.428496723279716>,0.05
    ,<-0.02121175241289921,-0.00046361164853532446,4.473784632748006>,0.05
    ,<-0.009259323036115978,-0.0006059902764170687,4.5223155498313705>,0.05
    ,<-0.0022663200484297928,-0.0007510051168014893,4.5718079992351255>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
