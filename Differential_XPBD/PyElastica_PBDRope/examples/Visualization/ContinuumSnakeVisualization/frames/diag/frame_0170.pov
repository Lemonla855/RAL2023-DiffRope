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
    ,<-0.21256157570683862,-0.0008778952586795397,4.20565784545249>,0.05
    ,<-0.1683861235823056,-0.0009600065599450725,4.229082772453734>,0.05
    ,<-0.1250842109465517,-0.0010299253539157492,4.254093231194229>,0.05
    ,<-0.0846595179884422,-0.0010653123165828525,4.283530503549249>,0.05
    ,<-0.05012978449712165,-0.0010452500991365134,4.319699499475049>,0.05
    ,<-0.02524915029615703,-0.0009632722345202741,4.363068027283105>,0.05
    ,<-0.013684354871951432,-0.000831497736071793,4.411703756258024>,0.05
    ,<-0.017905172428546244,-0.0006745201964801907,4.4615120436931255>,0.05
    ,<-0.03816091667590589,-0.0005232308194742594,4.507209555187771>,0.05
    ,<-0.07172711381899335,-0.0004068445047440357,4.5442530131308745>,0.05
    ,<-0.11363339362557705,-0.0003423961599375047,4.571519076613558>,0.05
    ,<-0.1590042095665226,-0.00032896092116314993,4.5925362942533265>,0.05
    ,<-0.20410254596491945,-0.0003516889123745659,4.614138067350787>,0.05
    ,<-0.2446440470035805,-0.00038740040163448195,4.643407857562333>,0.05
    ,<-0.2741458475032533,-0.00041315922240099147,4.683774434497439>,0.05
    ,<-0.2863570591622763,-0.00041887428531731065,4.73225197390091>,0.05
    ,<-0.2800223867287267,-0.00041217333960786277,4.781836351497583>,0.05
    ,<-0.25951691742725136,-0.0004124926783032567,4.827422746380084>,0.05
    ,<-0.23116977781353107,-0.0004320118074363185,4.868597274120526>,0.05
    ,<-0.20009106662017548,-0.0004674501445274983,4.9077565001302945>,0.05
    ,<-0.16905781344995507,-0.0005099232946249247,4.946956363679901>,0.05
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
