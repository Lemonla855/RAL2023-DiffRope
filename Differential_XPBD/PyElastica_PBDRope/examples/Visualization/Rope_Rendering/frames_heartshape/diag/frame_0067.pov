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
    ,<0.0,0.0,-30.0>,0.5562943816184998
    ,<-0.43058472871780396,1.2768112421035767,-28.522720336914062>,0.5418576002120972
    ,<-0.9263836145401001,2.4495785236358643,-26.98106575012207>,0.5480366945266724
    ,<-1.4519528150558472,3.755281686782837,-25.560924530029297>,0.5453417301177979
    ,<-2.172311305999756,4.20421028137207,-23.7508544921875>,0.5465095043182373
    ,<-2.505889415740967,6.100027561187744,-23.2096004486084>,0.5459869503974915
    ,<-3.1466002464294434,5.626959323883057,-21.375511169433594>,0.5461869835853577
    ,<-4.978241920471191,6.0455641746521,-20.68912696838379>,0.5460607409477234
    ,<-6.461769104003906,5.625961780548096,-19.414409637451172>,0.5461015105247498
    ,<-8.334065437316895,5.24458122253418,-18.821590423583984>,0.5460764765739441
    ,<-9.913848876953125,4.297399044036865,-18.04087257385254>,0.5460677146911621
    ,<-11.75861930847168,3.6882944107055664,-17.56126594543457>,0.546036958694458
    ,<-13.53046703338623,3.107800006866455,-16.834470748901367>,0.5460318922996521
    ,<-15.318074226379395,2.8543550968170166,-15.97238540649414>,0.5461077690124512
    ,<-17.19256019592285,2.6629436016082764,-15.306706428527832>,0.5460708737373352
    ,<-18.78583335876465,1.5809602737426758,-14.818544387817383>,0.549289345741272
    ,<-17.242149353027344,0.3482576012611389,-15.176609992980957>,0.5355034470558167
    ,<-15.846611022949219,0.9495108127593994,-13.400922775268555>,0.5077033638954163
    ,<-15.367294311523438,0.46127715706825256,-10.62094497680664>,0.4892551898956299
    ,<-14.518183708190918,0.061686038970947266,-7.425332069396973>,0.47820228338241577
    ,<-13.285057067871094,-0.11130549013614655,-4.00165319442749>,0.47221189737319946
    ,<-11.86577320098877,-0.14203952252864838,-0.4743386507034302>,0.47224515676498413
    ,<-10.39013385772705,-0.12224791198968887,3.08821177482605>,0.4713819921016693
    ,<-8.90803050994873,-0.09852767735719681,6.654729843139648>,0.47215938568115234
    ,<-7.532344341278076,-0.017753276973962784,10.247157096862793>,0.47224122285842896
    ,<-5.95165491104126,-0.053965210914611816,13.753328323364258>,0.47182419896125793
    ,<-4.46724271774292,-0.04000874608755112,17.313642501831055>,0.47204187512397766
    ,<-2.98100209236145,-0.027079883962869644,20.875473022460938>,0.4707167148590088
    ,<-1.4922131299972534,-0.012599540874361992,24.438007354736328>,0.4754757285118103
    ,<0.0,0.0,28.0>,0.4574834406375885
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
