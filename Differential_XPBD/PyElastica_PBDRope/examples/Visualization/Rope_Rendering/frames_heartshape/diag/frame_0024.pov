#include "../default.inc"

camera{
    location <40.0,100.5,-40.0>
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
    b_spline 30
    ,<0.0,0.0,-7.5>,0.20044134557247162
    ,<0.0007975111366249621,0.2972028851509094,-7.314144134521484>,0.197607159614563
    ,<0.001405258197337389,0.5926752090454102,-7.128774642944336>,0.19774723052978516
    ,<0.0016590674640610814,0.8914914727210999,-6.9402971267700195>,0.1977040022611618
    ,<0.0014326957752928138,1.184735655784607,-6.754467010498047>,0.19767269492149353
    ,<0.0006682872772216797,1.4867805242538452,-6.5633416175842285>,0.19762642681598663
    ,<-0.0004763461765833199,1.779447317123413,-6.380970001220703>,0.19758346676826477
    ,<-0.0017937622033059597,2.089949131011963,-6.195499897003174>,0.19756658375263214
    ,<-0.003045872785151005,2.389223337173462,-6.029811859130859>,0.19753526151180267
    ,<-0.0038150111213326454,2.715634822845459,-5.87094259262085>,0.197592630982399
    ,<-0.005503527820110321,3.03022837638855,-5.737546920776367>,0.19757212698459625
    ,<-0.009691119194030762,3.3665995597839355,-5.6145501136779785>,0.19770991802215576
    ,<-0.017723914235830307,3.6903021335601807,-5.4854912757873535>,0.19767028093338013
    ,<-0.02807130105793476,3.991440534591675,-5.315103054046631>,0.19759510457515717
    ,<-0.04903721809387207,4.24619722366333,-5.031314849853516>,0.19610987603664398
    ,<-0.04236757755279541,4.375263214111328,-4.682188510894775>,0.1932024359703064
    ,<-0.0924990177154541,4.394039630889893,-4.153632640838623>,0.18652555346488953
    ,<-0.06705769896507263,4.283346652984619,-3.6544950008392334>,0.17907404899597168
    ,<-0.06500042974948883,4.0401153564453125,-2.891055107116699>,0.1737619936466217
    ,<-0.061815641820430756,3.828017473220825,-2.297842025756836>,0.16974730789661407
    ,<-0.054685078561306,3.471036195755005,-1.3812216520309448>,0.1662830412387848
    ,<-0.049187466502189636,3.18939208984375,-0.6911879777908325>,0.16312341392040253
    ,<-0.04130878299474716,2.7607836723327637,0.3391282558441162>,0.1613817662000656
    ,<-0.035602010786533356,2.4323296546936035,1.1209288835525513>,0.1595786213874817
    ,<-0.028181226924061775,1.981481909751892,2.1945817470550537>,0.1591351181268692
    ,<-0.022649617865681648,1.6244927644729614,3.048790454864502>,0.15825262665748596
    ,<-0.016183234751224518,1.1845486164093018,4.1085591316223145>,0.15848590433597565
    ,<-0.010997286066412926,0.8091990947723389,5.02031946182251>,0.15829133987426758
    ,<-0.005484154913574457,0.39332154393196106,6.036154747009277>,0.15663808584213257
    ,<0.0,0.0,7.0>,0.17339670658111572
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }