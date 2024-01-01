#include "../default.inc"

camera{
    location <0,200,3>
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
    linear_spline 30
    ,<0.0,0.0,-30.0>,0.5494415163993835
    ,<-1.2741738557815552,0.929088294506073,-28.462465286254883>,0.528676450252533
    ,<-2.550116539001465,1.8604769706726074,-26.927749633789062>,0.5377950072288513
    ,<-3.8295211791992188,2.79632568359375,-25.398550033569336>,0.5336481332778931
    ,<-5.11394739151001,3.7385406494140625,-23.877330780029297>,0.53550785779953
    ,<-6.404792308807373,4.688657283782959,-22.3662166595459>,0.5346483588218689
    ,<-7.703279972076416,5.6477227210998535,-20.866952896118164>,0.5350104570388794
    ,<-9.010400772094727,6.616147994995117,-19.380956649780273>,0.5348248481750488
    ,<-10.326692581176758,7.5935139656066895,-17.909515380859375>,0.5349391102790833
    ,<-11.65173625946045,8.57834529876709,-16.453914642333984>,0.5350727438926697
    ,<-12.983341217041016,9.56802749633789,-15.014931678771973>,0.5354827642440796
    ,<-14.316393852233887,10.559208869934082,-13.591440200805664>,0.5361143350601196
    ,<-15.644325256347656,11.552940368652344,-12.176029205322266>,0.5366687774658203
    ,<-16.976613998413086,12.578187942504883,-10.750238418579102>,0.5345140099525452
    ,<-18.356815338134766,13.72467041015625,-9.298248291015625>,0.5267269015312195
    ,<-19.538246154785156,14.945249557495117,-7.475832939147949>,0.5101032853126526
    ,<-19.664836883544922,15.5849609375,-4.831035137176514>,0.48504194617271423
    ,<-19.471010208129883,15.779792785644531,-2.1357195377349854>,0.5492957234382629
    ,<-18.432418823242188,14.370511054992676,-2.7663004398345947>,0.5525485277175903
    ,<-16.589248657226562,12.751890182495117,-1.9928418397903442>,0.48615723848342896
    ,<-14.918014526367188,11.378488540649414,0.38122275471687317>,0.47299325466156006
    ,<-13.253134727478027,10.0449800491333,3.3834309577941895>,0.47322916984558105
    ,<-11.583333015441895,8.727643013000488,6.562777996063232>,0.4741629362106323
    ,<-9.924715995788574,7.439660549163818,9.702877044677734>,0.4769449532032013
    ,<-8.34470272064209,6.281830310821533,12.83998966217041>,0.4778901934623718
    ,<-6.631185531616211,4.9554829597473145,15.80260181427002>,0.4773499667644501
    ,<-4.980131149291992,3.7151756286621094,18.848222732543945>,0.477571964263916
    ,<-3.3240935802459717,2.476057767868042,21.897396087646484>,0.47581082582473755
    ,<-1.6643624305725098,1.2384759187698364,24.9493465423584>,0.4819292724132538
    ,<0.0,0.0,28.0>,0.45876047015190125
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }