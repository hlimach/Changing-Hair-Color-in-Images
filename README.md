# Changing-Hair-Color-in-Images
I've implemented HairColorGAN which uses a variation of the cycleGAN to change the hair color of a person in image, given just the target hair color. The model, in contrast to cycleGAN, uses only one generator and discriminator. The approach is inspired by [this](http://cs230.stanford.edu/projects_winter_2020/reports/32582032.pdf) student's project for the Stanford CS230 course.

## Output
The leftmost image shows the target hair color input into the networks, the middle image is the original image input, and the rightmost image is the output of the network.
![train_iter_3750](https://user-images.githubusercontent.com/57444629/215025727-648ed63a-784b-483b-8141-7422a2a026dd.png)
