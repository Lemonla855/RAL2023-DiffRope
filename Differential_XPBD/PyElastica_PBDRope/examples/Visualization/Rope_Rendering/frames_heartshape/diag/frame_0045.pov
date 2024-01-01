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
    linear_spline 30
    ,<0.0,0.0,-30.0>,0.5564764142036438
    ,<-0.12855516374111176,1.201248288154602,-28.398141860961914>,0.541822612285614
    ,<-0.2621329128742218,2.453521966934204,-26.836273193359375>,0.5481355786323547
    ,<-0.36690646409988403,3.731895685195923,-25.293527603149414>,0.5453550219535828
    ,<-0.7578719854354858,5.08811092376709,-23.867650985717773>,0.5465771555900574
    ,<-0.3612087368965149,6.3469696044921875,-22.35713768005371>,0.5461252331733704
    ,<-1.4634860754013062,7.965282917022705,-21.927841186523438>,0.5463733077049255
    ,<-1.8272638320922852,8.371332168579102,-19.99826431274414>,0.5462217330932617
    ,<-2.7932097911834717,10.118239402770996,-19.798805236816406>,0.5460590720176697
    ,<-3.5388078689575195,11.680620193481445,-18.77910614013672>,0.5462232828140259
    ,<-1.8435800075531006,12.74808120727539,-18.724746704101562>,0.5462599396705627
    ,<-2.8614773750305176,14.444962501525879,-19.061996459960938>,0.5461273789405823
    ,<-4.468214988708496,15.59144401550293,-19.42222785949707>,0.5461352467536926
    ,<-5.402095794677734,16.605384826660156,-20.880111694335938>,0.5465759634971619
    ,<-5.720012187957764,17.424633026123047,-19.085309982299805>,0.5462894439697266
    ,<-6.907598495483398,15.892666816711426,-19.589847564697266>,0.5474737882614136
    ,<-6.638695240020752,13.984153747558594,-20.083311080932617>,0.5445103049278259
    ,<-7.503036022186279,13.309404373168945,-17.948240280151367>,0.49645423889160156
    ,<-6.459077835083008,12.629744529724121,-14.991042137145996>,0.46589505672454834
    ,<-5.498371601104736,11.477457046508789,-11.42705249786377>,0.4660535156726837
    ,<-4.824458122253418,10.34138298034668,-7.489656925201416>,0.4622732996940613
    ,<-4.2717604637146,9.193379402160645,-3.5057663917541504>,0.46500638127326965
    ,<-3.7337357997894287,8.049653053283691,0.4480222463607788>,0.4646386206150055
    ,<-3.193909168243408,6.904866695404053,4.3847551345825195>,0.46494749188423157
    ,<-2.6564671993255615,5.7570977210998535,8.3186674118042>,0.46486103534698486
    ,<-2.122176170349121,4.607573986053467,12.253049850463867>,0.46480005979537964
    ,<-1.590099573135376,3.4569153785705566,16.188514709472656>,0.4649650454521179
    ,<-1.0594552755355835,2.3053083419799805,20.125150680541992>,0.46416065096855164
    ,<-0.5297164916992188,1.153056025505066,24.062524795532227>,0.46703624725341797
    ,<0.0,0.0,28.0>,0.45637375116348267
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
