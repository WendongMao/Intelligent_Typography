
'''
    def __getitem__1(self, index1):
        path1 = self.paths[index1 % self.size]
        path2 = self.paths[index1 % self.size]
        B_img = Image.open(path1).convert('RGB')
        A_img = Image.open(path2).convert('RGB')
        if self.opt.isTrain and not self.opt.no_flip:
            if random.random() > 0.5:
                B_img = B_img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                B_img = B_img
                
        w, h = B_img.size
        rw = random.randint(0, w - self.fineSize)
        rh = random.randint(0, h - self.fineSize)
        # print(rw, rh)
        B_img = B_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))

        w, h = B_img.size
        rw = random.randint(0, int(w/2))
        rh = random.randint(0, int(h/2))

        A_img = A_img.crop((int(rw/2),int(rh/2), int((rw + w)/2), int((rh + h)/2))
        #vutils.save_image(A_img,'training_A.png',normalize=True)
        #vutils.save_image(B_img,'training_B.png',normalize=True)
        #A_img = self.transform(A_img)
        B_img= self.transform(B_img)		
        return {'A': A_img, 'B': B_img,
                'A_paths': path1, 'B_paths': path2,
                'A_start_point':[(rw, rh)]}
'''