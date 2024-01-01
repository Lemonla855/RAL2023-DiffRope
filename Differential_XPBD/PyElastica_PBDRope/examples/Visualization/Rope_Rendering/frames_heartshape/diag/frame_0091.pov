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
    ,<0.0,0.0,-30.0>,0.5414551496505737
    ,<-1.011278748512268,1.2500743865966797,-28.145071029663086>,0.5146141052246094
    ,<-2.0224695205688477,2.5000643730163574,-26.290050506591797>,0.5266804099082947
    ,<-3.033493995666504,3.7499020099639893,-24.43486213684082>,0.5210495591163635
    ,<-4.044297218322754,4.999556064605713,-22.579421997070312>,0.5236423015594482
    ,<-5.054891586303711,6.249090194702148,-20.723590850830078>,0.522432804107666
    ,<-6.065377235412598,7.498691082000732,-18.867137908935547>,0.522949755191803
    ,<-7.075875282287598,8.748579025268555,-17.009857177734375>,0.5226405262947083
    ,<-8.086215019226074,9.99862003326416,-15.15211296081543>,0.5227405428886414
    ,<-9.0953369140625,11.247581481933594,-13.295877456665039>,0.5228694081306458
    ,<-10.10064697265625,12.492216110229492,-11.4459228515625>,0.5235317349433899
    ,<-11.09766960144043,13.726532936096191,-9.61058235168457>,0.5249102711677551
    ,<-12.087918281555176,14.950502395629883,-7.788599967956543>,0.5262008905410767
    ,<-13.10737133026123,16.198833465576172,-5.928295135498047>,0.5222026705741882
    ,<-14.220602989196777,17.55130386352539,-3.9017879962921143>,0.5069141983985901
    ,<-15.332136154174805,19.01702117919922,-1.5331878662109375>,0.484220027923584
    ,<-16.15196418762207,20.339092254638672,1.1883327960968018>,0.4879428446292877
    ,<-16.52614974975586,21.10390281677246,3.8905434608459473>,0.5385161638259888
    ,<-15.536202430725098,19.608217239379883,3.227627754211426>,0.5398907661437988
    ,<-14.14627456665039,17.568973541259766,3.6888954639434814>,0.49876320362091064
    ,<-12.76252555847168,15.772991180419922,5.61891508102417>,0.487269788980484
    ,<-11.373124122619629,14.050246238708496,8.047579765319824>,0.48712775111198425
    ,<-9.948373794555664,12.297649383544922,10.578741073608398>,0.4864867031574249
    ,<-8.514057159423828,10.531723976135254,13.090596199035645>,0.48770666122436523
    ,<-7.085718154907227,8.771419525146484,15.577934265136719>,0.48797833919525146
    ,<-5.665473937988281,7.026450157165527,18.06581687927246>,0.48787903785705566
    ,<-4.245451927185059,5.271966457366943,20.545881271362305>,0.48836907744407654
    ,<-2.8258559703826904,3.5167062282562256,23.027990341186523>,0.48606666922569275
    ,<-1.4051618576049805,1.7608026266098022,25.51096534729004>,0.4945897161960602
    ,<0.0,0.0,28.0>,0.4611254334449768
    texture{
        pigment{ color rgb<1.0,0.7215686274509804,0.18823529411764706> transmit 0.000000 }
        finish{ phong 1 }
    }
    }