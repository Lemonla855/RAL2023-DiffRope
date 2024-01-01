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
    ,<-0.17617477737469991,-0.0008693923016946159,3.116069538686602>,0.05
    ,<-0.13188669850684415,-0.0009594531870084828,3.139281323745337>,0.05
    ,<-0.08830224996665914,-0.0010395125730197326,3.1637975909558773>,0.05
    ,<-0.04724789435184425,-0.0010879897269175005,3.192354080459948>,0.05
    ,<-0.011538607593293186,-0.0010816853550649841,3.2273652886398088>,0.05
    ,<0.015217978095007807,-0.0010083769964331767,3.2696107504210348>,0.05
    ,<0.029291719311572574,-0.0008761975108148274,3.317589690257692>,0.05
    ,<0.027860554591106472,-0.0007097879076858654,3.367565113243539>,0.05
    ,<0.010125111014648773,-0.0005413499594558533,3.414306645244394>,0.05
    ,<-0.021683645893035963,-0.00040292624487134424,3.452875759214307>,0.05
    ,<-0.06271459083532928,-0.00031595252529200357,3.4814440375364826>,0.05
    ,<-0.10786879693651098,-0.0002839390627935475,3.5029202663784877>,0.05
    ,<-0.15327311112805825,-0.0002937167856034977,3.5238701489236903>,0.05
    ,<-0.19488507541727157,-0.00032237593890404546,3.55160107886536>,0.05
    ,<-0.226536900067677,-0.00034717013889451633,3.5903119173734566>,0.05
    ,<-0.24173755552081366,-0.000356304830640687,3.6379445297703086>,0.05
    ,<-0.238430039245704,-0.00035453077776058264,3.6878292463836653>,0.05
    ,<-0.22034800155581497,-0.00035922509617756235,3.7344353771880625>,0.05
    ,<-0.19371690108326278,-0.0003821569537951363,3.776743117080073>,0.05
    ,<-0.16385303915501406,-0.00042050236357230205,3.8168383249684954>,0.05
    ,<-0.13379480874656813,-0.0004657885553076102,3.856791774578534>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }