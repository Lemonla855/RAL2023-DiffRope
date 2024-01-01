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
    ,<0.0,0.0,0.0>,0.05
    ,<0.0,0.0,0.05>,0.05
    ,<0.0,0.0,0.1>,0.05
    ,<0.0,0.0,0.15000000000000002>,0.05
    ,<0.0,0.0,0.2>,0.05
    ,<0.0,0.0,0.25>,0.05
    ,<0.0,0.0,0.30000000000000004>,0.05
    ,<0.0,0.0,0.35000000000000003>,0.05
    ,<0.0,0.0,0.4>,0.05
    ,<0.0,0.0,0.45>,0.05
    ,<0.0,0.0,0.5>,0.05
    ,<0.0,0.0,0.55>,0.05
    ,<0.0,0.0,0.6000000000000001>,0.05
    ,<0.0,0.0,0.65>,0.05
    ,<0.0,0.0,0.7000000000000001>,0.05
    ,<0.0,0.0,0.75>,0.05
    ,<0.0,0.0,0.8>,0.05
    ,<0.0,0.0,0.8500000000000001>,0.05
    ,<0.0,0.0,0.9>,0.05
    ,<0.0,0.0,0.9500000000000001>,0.05
    ,<0.0,0.0,1.0>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
