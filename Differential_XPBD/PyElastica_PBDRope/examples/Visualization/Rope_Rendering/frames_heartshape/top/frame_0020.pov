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
    b_spline 30
    ,<0.0,0.0,-7.5>,0.2012917548418045
    ,<-0.013527456670999527,0.13103461265563965,-7.222644329071045>,0.19975142180919647
    ,<-0.026951372623443604,0.26242098212242126,-6.94460916519165>,0.19982105493545532
    ,<-0.04002920910716057,0.3930104672908783,-6.668307304382324>,0.19982755184173584
    ,<-0.05283660069108009,0.5244795083999634,-6.389995098114014>,0.1998458057641983
    ,<-0.0650687962770462,0.6538324952125549,-6.115548133850098>,0.1998796910047531
    ,<-0.07709172368049622,0.7839592099189758,-5.838016986846924>,0.19994908571243286
    ,<-0.0886695459485054,0.9098542928695679,-5.567143440246582>,0.20005232095718384
    ,<-0.10057999938726425,1.0361160039901733,-5.292540073394775>,0.20021992921829224
    ,<-0.11260969191789627,1.1568810939788818,-5.028729438781738>,0.20042310655117035
    ,<-0.1259460151195526,1.2834784984588623,-4.759984970092773>,0.20065030455589294
    ,<-0.13952147960662842,1.4134161472320557,-4.5123291015625>,0.2008177787065506
    ,<-0.1574394851922989,1.5826698541641235,-4.258990287780762>,0.20074251294136047
    ,<-0.18448612093925476,1.7622424364089966,-4.0472869873046875>,0.20031407475471497
    ,<-0.2265106737613678,1.9635398387908936,-3.759136199951172>,0.19886262714862823
    ,<-0.2564692497253418,2.053636074066162,-3.466355800628662>,0.19586454331874847
    ,<-0.2662478983402252,2.0753250122070312,-2.9616780281066895>,0.19204944372177124
    ,<-0.22281107306480408,2.055772066116333,-2.6136717796325684>,0.18621236085891724
    ,<-0.20532894134521484,1.9147281646728516,-1.8889379501342773>,0.17814595997333527
    ,<-0.18976323306560516,1.802887201309204,-1.3792016506195068>,0.17394234240055084
    ,<-0.16031518578529358,1.6015396118164062,-0.48646026849746704>,0.17070342600345612
    ,<-0.14034894108772278,1.4609018564224243,0.12682293355464935>,0.16782066226005554
    ,<-0.11024822294712067,1.2372400760650635,1.1104230880737305>,0.1668379306793213
    ,<-0.09106891602277756,1.0811200141906738,1.808923363685608>,0.16551901400089264
    ,<-0.06694584339857101,0.8654824495315552,2.7944436073303223>,0.16594979166984558
    ,<-0.05185987427830696,0.7028933167457581,3.5520741939544678>,0.16558292508125305
    ,<-0.03651706501841545,0.5123671889305115,4.489229679107666>,0.16646918654441833
    ,<-0.02513973042368889,0.3482896685600281,5.287446022033691>,0.1665002852678299
    ,<-0.012775646522641182,0.16961388289928436,6.166349411010742>,0.16580532491207123
    ,<0.0,0.0,7.0>,0.17882107198238373
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }