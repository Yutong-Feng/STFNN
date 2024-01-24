import torch as th
from torch import nn
from torch.cuda.amp.autocast_mode import autocast

from layers import Linear3D, get_non_linear, get_st_embedding
from utils import num_curl_loss


class FeatFusion(nn.Module):
    def __init__(self, args, device) -> None:
        super().__init__()
        self.non_linear = get_non_linear(args)
        self.st_embed = get_st_embedding(args, device)

        self.wind_dire_embed = nn.Embedding(26, args.wth_embed_dim)
        self.weather_embed = nn.Embedding(18, args.wth_embed_dim)
        self.hour_embed = nn.Embedding(25, args.time_embed_dim)
        self.weekday_embed = nn.Embedding(8, args.time_embed_dim)
        self.holiday_embed = nn.Embedding(3, args.time_embed_dim)

        in_dim = args.s_embed_dim + args.t_embed_dim + 10 +\
            2 * args.wth_embed_dim + 3 * args.time_embed_dim
        self.net = nn.Sequential(
            Linear3D(in_dim, args.hid_dim),
            self.non_linear
        )
        for _ in range(args.hid_layers):
            self.net.append(Linear3D(args.hid_dim, args.hid_dim))
            self.net.append(self.non_linear)

    def forward(self, st: th.Tensor, feat: th.Tensor):
        # st, feat -> hid_dim
        assert st.ndim == 3 and st.shape[2] == 3
        assert feat.ndim == 3 and feat.shape[2] == 15

        con_feat = feat[:, :, :10]
        wind_dire = feat[:, :, 10].long()
        weather = feat[:, :, 11].long()
        hour = feat[:, :, 12].long()
        wkday = feat[:, :, 13].long()
        is_holiday = feat[:, :, 14].long()

        st_code = self.st_embed(st)
        wind_dire_code = self.wind_dire_embed(wind_dire)
        wth_code = self.weather_embed(weather)
        hour_code = self.hour_embed(hour)
        wkday_code = self.weekday_embed(wkday)
        holiday_code = self.holiday_embed(is_holiday)
        input_tensor = th.concat(
            [st_code, con_feat, wind_dire_code, wth_code, hour_code, wkday_code, holiday_code], dim=2)
        result = self.net(input_tensor)
        return result


class RingEstimation(nn.Module):
    def __init__(
        self,
        args,
        device,
    ) -> None:
        super().__init__()

        self.non_linear = get_non_linear(args)
        self.st_embed = get_st_embedding(args, device)
        self.step_embed = nn.Embedding(args.max_step + 1, args.step_embed_dim)

        concat_dim = args.s_embed_dim + args.t_embed_dim
        self.net = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=args.step_embed_dim + concat_dim,
                dim_feedforward=args.dim_feed,
                nhead=args.nhead,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=args.ring_layers,
        )
        self.linear = Linear3D(concat_dim + args.step_embed_dim, 3)

    def forward(self, memory: th.Tensor, end_st: th.Tensor, step: th.Tensor):
        assert memory.shape == end_st.shape, "memory.shape {memory.shape} != \
            end_st.shape{end_st.shape}"
        B, L, C = memory.shape
        assert C == 3, "C = {C} is not 3"

        step_embed = self.step_embed(step).reshape(1, 1, -1).repeat(B, L, 1)
        memory_cat, end_embed_cat = [
            th.concat([self.st_embed(st), step_embed], dim=2)
            for st in (memory, end_st)
        ]
        end_repr = self.net(end_embed_cat, memory_cat)
        output = self.non_linear(end_repr)
        output = self.linear(output)
        return output


class STAttention(nn.Module):
    def __init__(self, args, device) -> None:
        super().__init__()
        self.read_out = FeatFusion(args, device)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args.hid_dim,
                dim_feedforward=args.dim_feed,
                nhead=args.nhead,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=args.weight_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=args.hid_dim,
                dim_feedforward=args.dim_feed,
                nhead=args.nhead,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=args.weight_layers,
        )
        self.linear = Linear3D(args.hid_dim, 1)

    def forward(self,
                query_st: th.Tensor,
                query_feat: th.Tensor,
                ngb_st: th.Tensor,
                ngb_feat: th.Tensor):
        neighbor_code = self.read_out(ngb_st, ngb_feat)
        query_code = self.read_out(query_st, query_feat)

        query_repr = self.encoder(query_code)
        neighbor_repr = self.decoder(neighbor_code, query_repr)

        result = self.linear(neighbor_repr).squeeze(2).softmax(dim=1)
        return result


class PyramidalInference(nn.Module):
    def __init__(
        self,
        args,
        device
    ) -> None:
        super().__init__()
        self.max_step = args.max_step
        self.device = device

        self.st_embed = get_st_embedding(args, device)
        self.read_out_f = FeatFusion(args, device)
        self.read_out_l = Linear3D(args.hid_dim, 3)
        self.read_in_f = FeatFusion(args, device)
        self.read_in_l = Linear3D(args.hid_dim, 3)

        self.ring_est = RingEstimation(args, device)
        self.dym_graph = args.dym_graph
        if args.dym_graph:
            self.st_weight = STAttention(args, device)

    def pyramidal_aggregation(self, start_st, start_grad, final_st):
        con_grad_buffer = [start_grad]
        dire_vec = (final_st - start_st) / self.max_step
        dire_vec_norm = th.einsum(
            "blc,bl->blc", dire_vec, th.reciprocal(th.norm(dire_vec, p=2, dim=2)
                                                   ))
        for step in range(self.max_step):
            memory = con_grad_buffer[-1]
            end_st = start_st + dire_vec
            con_grad = self.ring_est(
                memory, end_st, th.tensor(step).to(self.device))
            con_grad_buffer.append(con_grad)
            start_st = end_st
        con_grad_buffer = th.stack(con_grad_buffer)
        cum_grad = con_grad_buffer.sum(dim=0)
        norm_grad = th.einsum("blc,blc->blc", cum_grad, dire_vec_norm)

        # curl
        grad_diff = con_grad_buffer[1:] - con_grad_buffer[:-1]
        jcb_mtx = th.einsum("sblc,sbld->sblcd", grad_diff,
                            th.reciprocal(dire_vec).expand(self.max_step, -1, -1, -1))
        curl_loss = num_curl_loss(jcb_mtx)

        return norm_grad, curl_loss

    def idw_ses(self, dist_mtx: th.Tensor):
        B, L, C = dist_mtx.shape
        s_weight = th.einsum("blc,bl->blc", dist_mtx,
                             dist_mtx.sum(dim=-1).reciprocal())
        t_weight = th.cumprod(
            th.tensor([[0.5]]).repeat(L, C), 0).to(self.device)
        weight = th.einsum("blc,lc->blc", s_weight, t_weight).reshape(B, -1)
        return weight

    def forward(self,
                query_st: th.Tensor,
                query_feat: th.Tensor,
                ngb_st: th.Tensor,
                ngb_feat: th.Tensor,
                ngb_label: th.Tensor,
                dist: th.Tensor):

        query_st = query_st.unsqueeze(1).repeat(1, ngb_st.shape[1], 1)
        query_feat = query_feat.unsqueeze(dim=1).repeat(
            1, ngb_feat.shape[1], 1
        )

        start_st = ngb_st.clone()
        final_st = query_st.clone()

        start_grad = self.read_out_l(self.read_out_f(ngb_st, ngb_feat))
        norm_grad, curl_loss = self.pyramidal_aggregation(
            start_st, start_grad, final_st)
        end_grad = self.read_in_l(self.read_in_f(norm_grad, query_feat))

        est_labels = end_grad.sum(dim=2)
        if self.dym_graph:
            st_weight = self.st_weight(query_st, query_feat, ngb_st, ngb_feat)
        else:
            st_weight = self.idw_ses(dist)
        result = th.einsum("bl,bl -> b", (ngb_label + est_labels), st_weight)
        return result, curl_loss


class STNeVF(nn.Module):
    def __init__(self, args, device) -> None:
        super().__init__()
        self.backbone = PyramidalInference(args, device)

    def forward(self, query_info: th.Tensor, ngb_data: th.Tensor, dist: th.Tensor):
        query_st = query_info[:, :3]
        query_feat = query_info[:, 3:]
        ngb_st = ngb_data[:, :, :3]
        ngb_feat = ngb_data[:, :, 3:-1]
        ngb_label = ngb_data[:, :, -1]
        return self.backbone(query_st, query_feat, ngb_st, ngb_feat, ngb_label, dist)


def get_model(args, device):
    return STNeVF(args, device).to(device)


def predict(batch_data, model, device, mm_size=0, training=True):
    if not mm_size:
        query_data = batch_data["stat"].float().to(
            device, non_blocking=training)
        ngb_data = batch_data["ngb"].float().to(device, non_blocking=training)
        mask = batch_data["mask_index"]
        data_index = batch_data["data_index"]
        dist = batch_data["dist"].float().to(
            device, non_blocking=training)

        query_info = query_data[:, :-1]
        query_label = query_data[:, -1]
        with autocast():
            data_recover, curl_loss = model(query_info, ngb_data, dist)
        return data_recover, curl_loss, query_label, data_index, mask
    else:
        batch_len = len(batch_data["data_index"])
        data_recover = []
        curl_loss = []
        query_label = []
        dist = []
        mask = batch_data["mask_index"]
        data_index = batch_data["data_index"]
        for i in range(int(batch_len/mm_size)+1):
            left_index = int(i*mm_size)
            right_index = int(min((i+1)*mm_size, batch_len))
            if left_index == right_index:
                break
            query_data = batch_data["stat"].float()[left_index:right_index].to(
                device, non_blocking=training)
            ngb_data = batch_data["ngb"].float()[left_index:right_index].to(
                device, non_blocking=training)
            dist = batch_data["dist"].float()[left_index:right_index].to(
                device, non_blocking=training)

            query_info = query_data[:, :-1]
            m_query_label = query_data[:, -1]
            with autocast():
                m_data_recover, m_curl_loss = model(query_info, ngb_data, dist)
            data_recover.append(m_data_recover)
            curl_loss.append(m_curl_loss)
            query_label.append(m_query_label)
        data_recover = th.concat(data_recover)
        curl_loss = th.stack(curl_loss).mean()
        query_label = th.concat(query_label)
        return data_recover, curl_loss, query_label, data_index, mask


if __name__ == "__main__":
    from config import ConfigFactory
    from dataset import get_air_loader_normalizer

    args, msg = ConfigFactory.build()
    print(msg)
    # DEVICE = "cuda" if th.cuda.is_available() else "cpu"
    DEVICE = "cpu"
    MODEL = get_model(args, DEVICE)
    criterion = nn.MSELoss()
    (train_loader, val_loader, test_loader), dm = get_air_loader_normalizer(
        args
    )
    batch_data = next(iter(train_loader))
    data_recover, curl_loss, query_label, data_index, mask = predict(
        batch_data, MODEL, DEVICE, mm_size=args.mm_size)
    loss = criterion(data_recover, query_label)
    pred_label = dm.denorm(
        data_recover.cpu().detach().numpy(), data_index
    )
    label = dm.denorm(
        query_label.cpu().detach().numpy(), data_index)
    print(f"Loss: {loss.item()}")
    print("Finish!")
