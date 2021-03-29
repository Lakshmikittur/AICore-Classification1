# import torch


# def get(args):
#     if args.model == "baseline":
#       return Baseline(args)
#     else :
#       return Leaky(args)
#     # elif args.model == "relumore": #gave lesser than relu less
#     #   return ReluMore(args)
#     # elif args.model == "reluless":
#     #   return ReluLess(args)




# # 25%
# class Baseline(torch.nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.layer = torch.nn.Linear(100, args.num_classes)

#     def forward(self, inputs):
#         return self.layer(inputs)


# # 95% train, 70% on validation
# class Leaky(torch.nn.Module):
#   def __init__(self, args):
#     # float, half
#     super().__init__()
#     self.model = torch.nn.Sequential(
#         torch.nn.Linear(100, 512),
#         torch.nn.LeakyReLU(0.1),
#         torch.nn.BatchNorm1d(512),
#         torch.nn.Linear(512, 256),
#         torch.nn.LeakyReLU(0.1),
#         torch.nn.BatchNorm1d(256),
#         torch.nn.Linear(256, 128),
#         torch.nn.LeakyReLU(0.1),
#         torch.nn.BatchNorm1d(128),
#         torch.nn.Linear(128, 64),
#         torch.nn.LeakyReLU(0.1),
#         torch.nn.BatchNorm1d(64),
#         torch.nn.Linear(64, 32),
#         torch.nn.LeakyReLU(0.1),
#         torch.nn.BatchNorm1d(32),
#         torch.nn.Linear(32, args.num_classes),
#     )

# # class ReluMore(torch.nn.Module):
# #   def __init__(self, args):
# #       # float, half
# #     super().__init__()
# #     self.model = torch.nn.Sequential(
# #           torch.nn.Linear(100, 512),
# #           torch.nn.ReLU(),
# #           torch.nn.BatchNorm1d(512),
# #           torch.nn.Linear(512, 256),
# #           torch.nn.ReLU(),
# #           torch.nn.BatchNorm1d(256),
# #           torch.nn.Linear(256, 128),
# #           torch.nn.ReLU(),
# #           torch.nn.BatchNorm1d(128),
# #           torch.nn.Linear(128, 64),
# #           torch.nn.ReLU(),
# #           torch.nn.BatchNorm1d(64),
# #           torch.nn.Linear(64, 32),
# #           torch.nn.ReLU(),
# #           torch.nn.BatchNorm1d(32),
# #           torch.nn.Linear(32, args.num_classes),
# #       )
# #   class ReluLess(torch.nn.Module):
# #     def __init__(self, args):
# #         # float, half
# #       super().__init__()
# #       self.model = torch.nn.Sequential(
# #           torch.nn.Linear(100, 512),
# #           torch.nn.ReLU(),
# #           torch.nn.BatchNorm1d(512),
# #           torch.nn.Linear(512, 128),
# #           torch.nn.ReLU(),
# #           torch.nn.BatchNorm1d(128),
# #           torch.nn.Linear(128, 32),
# #           torch.nn.ReLU(),
# #           torch.nn.BatchNorm1d(32),
# #           torch.nn.Linear(32, args.num_classes),
# #       )
    

#     def forward(self, inputs):
#         return self.model(inputs)
import torch


def get(args):
    if args.model == "baseline":
        return Baseline(args)
    else:
        return Serious(args)


# 25%
class Baseline(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer = torch.nn.Linear(100, args.num_classes)

    def forward(self, inputs):
        return self.layer(inputs)



class Serious(torch.nn.Module):
  def __init__(self, args):
    # float, half
    super().__init__()
    self.model = torch.nn.Sequential(
        torch.nn.Linear(100, 512),
        torch.nn.LeakyReLU(0.1),
        torch.nn.BatchNorm1d(512),
        torch.nn.Linear(512, 256),
        torch.nn.LeakyReLU(0.1),
        torch.nn.BatchNorm1d(256),
        # torch.nn.Linear(256, 128),
        # torch.nn.LeakyReLU(0.1),
        # torch.nn.BatchNorm1d(128),
        # torch.nn.Linear(128, 64),
        # torch.nn.LeakyReLU(0.1),
        # torch.nn.BatchNorm1d(64),
        # torch.nn.Linear(256, 64),
        # torch.nn.LeakyReLU(0.1),
        # torch.nn.BatchNorm1d(64),
        torch.nn.Linear(256, args.num_classes),
    )


  def forward(self, inputs):
    return self.model(inputs)

