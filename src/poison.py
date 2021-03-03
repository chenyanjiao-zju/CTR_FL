
def get_poison_batch(self, adversarial_index=-1):
    images, targets = bptt

    poison_count = 0
    new_images = images
    new_targets = targets


    #
    # new_images = new_images.to(device)
    # new_targets = new_targets.to(device).long()


    return new_images, new_targets, poison_count


def add_pixel_pattern(self, ori_image, adversarial_index):
    image = copy.deepcopy(ori_image)
    trigger = self.global_trigger
    poison_patterns = []
    if adversarial_index == -1:
        for i in range(0, self.params['trigger_num']):
            poison_patterns = poison_patterns + self.params[str(i) + '_poison_pattern']
    else:
        poison_patterns = self.params[str(adversarial_index) + '_poison_pattern']
    for i in range(0, len(poison_patterns)):
        pos = poison_patterns[i]
        image[0][pos[0]][pos[1]] = trigger[0][pos[0]][pos[1]]
        image[1][pos[0]][pos[1]] = trigger[1][pos[0]][pos[1]]
        image[2][pos[0]][pos[1]] = trigger[2][pos[0]][pos[1]]

    return image