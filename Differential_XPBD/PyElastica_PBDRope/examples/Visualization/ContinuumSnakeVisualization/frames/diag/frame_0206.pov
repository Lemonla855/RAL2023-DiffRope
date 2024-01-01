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
    ,<-0.34503832993763817,-0.0018558038302274758,5.175107744352109>,0.05
    ,<-0.3197670375390923,-0.0015512288810330893,5.218230458700995>,0.05
    ,<-0.2907247303722623,-0.0012584434953864406,5.258916386202967>,0.05
    ,<-0.2570726139665521,-0.0009929601894564989,5.295890724937391>,0.05
    ,<-0.21981462494894405,-0.0007719755676302877,5.329241018244288>,0.05
    ,<-0.1806523497312361,-0.0006039847046894068,5.36034488860153>,0.05
    ,<-0.1414801066680242,-0.0004820128079821819,5.391443581193178>,0.05
    ,<-0.10456495369030262,-0.000386706819035293,5.425190991271>,0.05
    ,<-0.0732656137417204,-0.00029346607861726855,5.46419659949962>,0.05
    ,<-0.052630577139009545,-0.00018440735677384835,5.509739730575072>,0.05
    ,<-0.0483395821783222,-6.437389061898758e-05,5.5595403114616655>,0.05
    ,<-0.06332082309817824,4.0184804024022443e-05,5.607216555399561>,0.05
    ,<-0.09484215014733452,0.0001001453780123625,5.645996353567586>,0.05
    ,<-0.13624261491892387,9.900133178345222e-05,5.674000516253341>,0.05
    ,<-0.1813806976147054,3.874983703476862e-05,5.695486458499888>,0.05
    ,<-0.22618329014574592,-6.460192408084219e-05,5.717671386479989>,0.05
    ,<-0.2669215177655276,-0.0001899014325919302,5.746647033516415>,0.05
    ,<-0.2995011152246263,-0.00031949626378541837,5.784558390250742>,0.05
    ,<-0.32157304238831685,-0.0004453678144918691,5.829404978984574>,0.05
    ,<-0.3344965689553003,-0.0005696904261261173,5.877688767409787>,0.05
    ,<-0.3425973367924632,-0.0006952522744044405,5.927014408172707>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
