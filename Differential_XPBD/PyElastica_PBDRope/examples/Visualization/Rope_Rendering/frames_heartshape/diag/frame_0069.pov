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
    ,<0.0,0.0,-30.0>,0.5558968186378479
    ,<-0.4772752523422241,1.1797435283660889,-28.443988800048828>,0.5410836935043335
    ,<-1.205074429512024,2.310901403427124,-26.950233459472656>,0.5474118590354919
    ,<-1.5681408643722534,3.003561496734619,-25.097970962524414>,0.544616162776947
    ,<-3.045656204223633,4.085282325744629,-24.27011489868164>,0.5460825562477112
    ,<-2.8157553672790527,5.26277494430542,-22.663312911987305>,0.545600414276123
    ,<-4.68474006652832,5.281771183013916,-21.934574127197266>,0.5459228157997131
    ,<-5.6645894050598145,5.738001346588135,-20.24897003173828>,0.5459982752799988
    ,<-7.466672420501709,4.95139741897583,-19.873260498046875>,0.5459741353988647
    ,<-8.933834075927734,4.946026802062988,-18.510631561279297>,0.545905351638794
    ,<-10.738758087158203,4.104909420013428,-18.289785385131836>,0.5457524657249451
    ,<-12.248589515686035,4.0547943115234375,-16.971513748168945>,0.5456984043121338
    ,<-14.054156303405762,3.725602388381958,-16.16922950744629>,0.5459730625152588
    ,<-15.753175735473633,3.285987615585327,-15.220975875854492>,0.5469940900802612
    ,<-17.45496940612793,2.8465261459350586,-14.309919357299805>,0.547141432762146
    ,<-19.215744018554688,3.2391786575317383,-13.532628059387207>,0.5541222095489502
    ,<-19.65484046936035,1.3600538969039917,-13.927481651306152>,0.5334581732749939
    ,<-17.94962501525879,0.675059974193573,-12.51496410369873>,0.5099924206733704
    ,<-17.39447593688965,1.1995997428894043,-9.814399719238281>,0.4930872321128845
    ,<-16.472257614135742,1.451693058013916,-6.660789966583252>,0.4766395688056946
    ,<-15.124322891235352,1.4458502531051636,-3.280200242996216>,0.4720716178417206
    ,<-13.527076721191406,1.304800033569336,0.18556857109069824>,0.471554160118103
    ,<-11.805806159973145,1.1529258489608765,3.6520981788635254>,0.4712516665458679
    ,<-10.095459938049316,0.9872097373008728,7.1232500076293945>,0.471723735332489
    ,<-8.401884078979492,0.824059784412384,10.594144821166992>,0.4716831147670746
    ,<-6.718660354614258,0.6614980101585388,14.068659782409668>,0.47152578830718994
    ,<-5.0388665199279785,0.4978042542934418,17.54776382446289>,0.47176864743232727
    ,<-3.3596653938293457,0.33281874656677246,21.030370712280273>,0.4704311192035675
    ,<-1.6803656816482544,0.16700229048728943,24.514997482299805>,0.47513052821159363
    ,<0.0,0.0,28.0>,0.4574340581893921
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
