"""Paired first-seed comparison; resample original IDs jointly across noise seeds."""
import json,re,base64
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from markdown_it import MarkdownIt
from vihsd.data import dump_json,load_raw,normalize_text,sha256_bytes

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/robustness';OUT.mkdir(parents=True,exist_ok=True)


def macro_from_predictions(y,p,indices):
    cm=np.bincount((3*y[indices]+p[indices]),minlength=9).reshape(3,3)
    denom=cm.sum(0)+cm.sum(1)
    return float(np.divide(2*np.diag(cm),denom,out=np.zeros(3,dtype=float),where=denom>0).mean())


def run(roots):
    rows=[];predictions={};summaries={};banks=[];view_hashes={}
    for condition,path in roots.items():
        root=Path(path)
        done=json.loads((root/'ROBUSTNESS_COMPLETE.json').read_text());banks.append(done['bank_sha256'])
        for view_path in sorted((root/'data/noise_views').glob('*/view.parquet')):
            view=pd.read_parquet(view_path)
            digest=sha256_bytes(pd.util.hash_pandas_object(view[['sample_id','text_noisy','changed','actual_rate','edits']],index=False).to_numpy().tobytes())
            name=view_path.parent.name
            if name in view_hashes:assert view_hashes[name]==digest,f'Different noise view: {name}'
            else:view_hashes[name]=digest
        assert done.get('noise_version')=='human-dual-source-calibrated-v2','Only v2 belongs in this comparison'
        for model in [condition,'baseline']+done.get('extra_references',[]):
            data=json.loads((root/'robustness'/model/'summary.json').read_text())
            if model=='baseline' and model in summaries:
                # Ensure both Kaggle jobs evaluated exactly the same frozen reference.
                for profile in data:
                    assert abs(data[profile]['macro_f1']-summaries[model][profile]['macro_f1'])<1e-12
                    repeated=pd.read_parquet(root/'robustness'/model/f'{profile}_predictions.parquet').sort_values('sample_id')
                    previous=predictions[model,profile]
                    assert repeated[['sample_id','label','prediction']].equals(previous[['sample_id','label','prediction']]),f'Baseline predictions differ: {profile}'
                    assert np.allclose(repeated[['p_clean','p_offensive','p_hate']],previous[['p_clean','p_offensive','p_hate']],atol=1e-6,rtol=1e-5)
                continue
            summaries[model]=data
            for profile,metrics in data.items():
                p=pd.read_parquet(root/'robustness'/model/f'{profile}_predictions.parquet').sort_values('sample_id')
                predictions[model,profile]=p
                rows.append({'model':model,'profile':profile,'macro_f1':metrics['macro_f1'],'accuracy':metrics['accuracy'],
                             'coverage':metrics.get('coverage'),'actual_rate_mean':metrics.get('actual_rate_mean'),
                             'changed_only_f1':metrics.get('changed_only',{}).get('macro_f1'),
                             'original_on_changed_f1':metrics.get('changed_only_original_macro_f1')})
    assert len(set(banks))==1,'Noise banks differ between model runs'
    assert len(view_hashes)==10,'All ten shared noisy profiles required'
    dump_json(OUT/'SHARED_BENCHMARK_VERIFICATION.json',{'passed':True,'noise_version':'human-dual-source-calibrated-v2','bank_sha256':banks[0],'decoded_view_hashes':view_hashes,'baseline_predictions_identical_across_jobs':True})
    table=pd.DataFrame(rows);table.to_csv(OUT/'all_results.csv',index=False)
    base=predictions['baseline','original'];y=base.label.to_numpy();ids=base.sample_id.to_numpy()
    for key,p in predictions.items():assert np.array_equal(p.sample_id,ids) and np.array_equal(p.label,y)
    grouped=[]
    for model in summaries:
        for category,pattern in [('original',r'^original$'),('empirical_seen',r'^empirical_train_s'),('empirical_unseen',r'^empirical_unseen_s')]:
            sub=table[(table.model==model)&table.profile.str.contains(pattern,regex=True)]
            grouped.append({'model':model,'category':category,'mean_macro_f1':float(sub.macro_f1.mean()),'noise_seed_std':float(sub.macro_f1.std(ddof=0)),'mean_coverage':float(sub.coverage.mean()) if sub.coverage.notna().any() else 0})
    means=pd.DataFrame(grouped);means.to_csv(OUT/'primary_summary.csv',index=False)
    # Keep duplicate original texts and all noise variants together in bootstrap.
    original=load_raw(ROOT)['test'].set_index('sample_id').loc[ids]
    groups={}
    for i,text in enumerate(original.text_raw):
        groups.setdefault(sha256_bytes(normalize_text(text).encode()),[]).append(i)
    members=[np.asarray(v,dtype=np.int32) for v in groups.values()]
    rng=np.random.default_rng(8741)
    sample_indices=[np.concatenate([members[i] for i in rng.integers(0,len(members),len(members))]) for _ in range(500)]
    bootstrap=[]
    for model in summaries:
        if model=='baseline':continue
        for category,prefix in [('original','original'),('empirical_seen','empirical_train_s'),('empirical_unseen','empirical_unseen_s')]:
            profiles=[p for p in summaries[model] if p==prefix or p.startswith(prefix) and prefix!='original']
            differences=[]
            for indices in sample_indices:
                proposed=np.mean([macro_from_predictions(y,predictions[model,p].prediction.to_numpy(),indices) for p in profiles])
                reference=np.mean([macro_from_predictions(y,predictions['baseline',p].prediction.to_numpy(),indices) for p in profiles])
                differences.append(proposed-reference)
            point=float(means.query('model == @model and category == @category').mean_macro_f1.iloc[0]-means.query('model == "baseline" and category == @category').mean_macro_f1.iloc[0])
            bootstrap.append({'model':model,'category':category,'delta_macro_f1':point,'ci95_low':float(np.quantile(differences,.025)),'ci95_high':float(np.quantile(differences,.975)),'bootstrap_replicates':500,'resampling_unit':'normalized original-text group jointly across noise seeds','training_seeds':1})
    dump_json(OUT/'paired_bootstrap.json',bootstrap)
    sns.set_theme(style='whitegrid',font='DejaVu Sans')
    display_names={'baseline':'Baseline','augmentation-v2':'Augmentation v2','fgm-small':'FGM ε=0.1','fgm-eps1-reference':'FGM ε=1.0'}
    palette=dict(zip(display_names.values(),sns.color_palette('colorblind',4)))
    plotted=means.assign(model=means.model.map(display_names))
    fig,ax=plt.subplots(figsize=(11,4.8));sns.barplot(data=plotted,x='category',y='mean_macro_f1',hue='model',errorbar=None,ax=ax,palette=palette,saturation=1);ax.set(title='PhoBERT: test gốc và biến thể có bằng chứng corpus',xlabel='',ylabel='Macro-F1',ylim=(0,1));ax.legend(title='Model',loc='upper right');fig.tight_layout();fig.savefig(OUT/'01_primary_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for model in summaries:
        stress=table[(table.model==model)&table.profile.str.startswith('stress_')].copy();stress['requested']=stress.profile.str.extract(r'stress_(\d+)').astype(int)
        stress=stress.sort_values('requested')
        axes[0].plot(stress.requested,stress.macro_f1,marker='o',label=display_names.get(model,model),color=palette[display_names.get(model,model)])
        if model=='baseline':axes[1].plot(stress.requested,stress.actual_rate_mean*100,marker='o',label='Cùng noise views cho mọi model')
    axes[0].set(title='Controlled stress (mức yêu cầu)',xlabel='Mức thêm nhiễu yêu cầu (%)',ylabel='Macro-F1');axes[0].legend()
    axes[1].set(title='Mức thay đổi thực tế — phải đọc cùng điểm F1',xlabel='Mức yêu cầu (%)',ylabel='Tỷ lệ đơn vị nguồn bị sửa (%)');axes[1].legend();fig.tight_layout();fig.savefig(OUT/'02_stress_and_actual_rates.png',dpi=160);plt.close(fig)
    text='''# Kết quả first-seed robustness: nguồn human-normalization → training

## Thiết kế đã chạy

Baseline PhoBERT khóa ở notebook version 2. Hai model mới được khởi tạo lại từ cùng pretrained revision, seed 42, 5 epoch; best checkpoint chọn bằng clean validation Macro-F1. FGM thử hai epsilon 1.0 và 0.1; candidate 0.1 được bổ sung dựa trên validation trước khi thấy test. Chọn epsilon bằng clean validation Macro-F1, không dùng test; chưa chạy grid đầy đủ hoặc nhiều training seeds. Augmentation chọn thử 50% comment/epoch, chỉ dùng train bank; số mẫu và optimizer steps giữ nguyên.

Biến thể lấy từ ViLexNorm TRAIN và Small-LexNormViHSD released TRAIN có human normalization annotation. Trọng số conditional được hiệu chỉnh theo loại edit bằng TRAIN. Test sử dụng cùng noise bank v2 và seed cho mọi model. Bank unseen không có surface trong train bank hoặc raw ViHSD train; không có nghĩa chưa xuất hiện trong pretraining. FGM epsilon1 được đánh giá lại từ checkpoint pilot trên cùng v2 views; không trộn điểm benchmark v1 với v2.

**Giới hạn:** nhãn toxic của câu tổng hợp là giả định kế thừa có lọc, chưa được người gán lại. ViLexNorm không cung cấp gold toxic labels cho các câu mới. Empirical-event profile thêm một sự kiện lexical; không tuyên bố mô phỏng đầy đủ phân phối UGC tự nhiên. Xem báo cáo nghiên cứu noise để đọc selection/domain bias và mapping phụ thuộc ngữ cảnh.

## Kết quả chính

Trung bình trên 3 seed noise cho empirical profiles; toàn bộ là một training seed.

| Model | Test gốc | Empirical seen | Empirical unseen |
|---|---:|---:|---:|
'''
    for model in summaries:
        values=means[means.model==model].set_index('category').mean_macro_f1
        text+=f"| {model} | {values['original']:.4f} | {values['empirical_seen']:.4f} | {values['empirical_unseen']:.4f} |\n"
    aug_seen=next(r for r in bootstrap if r['model']=='augmentation-v2' and r['category']=='empirical_seen')
    fgm_unseen=next(r for r in bootstrap if r['model']=='fgm-small' and r['category']=='empirical_unseen')
    text+=f"\nSeed 42: augmentation v2 thay đổi {100*aug_seen['delta_macro_f1']:+.2f} điểm phần trăm Macro-F1 trên empirical seen; FGM ε=0,1 thay đổi {100*fgm_unseen['delta_macro_f1']:+.2f} điểm trên empirical unseen so baseline. CI bootstrap có điều kiện trên checkpoint cố định, chưa điều chỉnh nhiều phép so sánh và không đo bất định qua training seeds.\n"
    if all(r['ci95_low']<=0<=r['ci95_high'] for r in bootstrap if r['category']=='original'):
        text+='CI chênh lệch test gốc đều cắt 0; chưa có bằng chứng chắc về cải thiện clean performance.\n'
    shared_file=OUT/'shared_eligible_comparison.csv'
    if shared_file.exists():
        common=pd.read_csv(shared_file)
        def common_score(model):return float(common.query('model == @model and category == "empirical_unseen"').mean_macro_f1.iloc[0])
        n_common=int(common.n_same_original_ids.iloc[0])
        aug_value=common_score('augmentation-v2');base_value=common_score('baseline');fgm_value=common_score('fgm-small')
        direction='không cải thiện' if aug_value<=base_value else 'cao hơn baseline'
        text+=f"\nCùng {n_common:,} original IDs đủ điều kiện trong cả sáu empirical views: augmentation v2 {direction} unseen ({aug_value:.4f} so {base_value:.4f}); FGM ε=0,1 đạt {fgm_value:.4f}. Không diễn giải điểm toàn test như cải thiện đồng đều trên các câu bị sửa.\n"
    stress_aug=table.query('model == "augmentation-v2" and profile == "stress_50_s201"').iloc[0]
    stress_base=table.query('model == "baseline" and profile == "stress_50_s201"').iloc[0]
    text+=f"\nControlled stress 50: augmentation v2 {stress_aug.macro_f1:.4f}, baseline {stress_base.macro_f1:.4f}; tỷ lệ sửa thực tế {stress_aug.actual_rate_mean:.2%}, không phải 50% toàn bộ đơn vị test.\n"
    candidates=[]
    if 'fgm-eps1-reference' in summaries:
        refroot=next(Path(path) for path in roots.values() if (Path(path)/'robustness/fgm-eps1-reference/selection.json').exists())
        selection=json.loads((refroot/'robustness/fgm-eps1-reference/selection.json').read_text())
        candidates.append({'condition':'fgm-eps1-reference','validation_macro_f1':selection['score'],'epsilon':1.0})
    for condition in ['fgm-small']:
        if condition in roots:
            complete=json.loads((Path(roots[condition])/'ROBUSTNESS_COMPLETE.json').read_text())
            selection=json.loads((Path(roots[condition])/'runs'/complete['run_name']/'best/selection.json').read_text())
            candidates.append({'condition':condition,'validation_macro_f1':selection['score'],'epsilon':.1 if condition=='fgm-small' else 1.0})
    if candidates:
        selected=max(candidates,key=lambda c:(c['validation_macro_f1'],-c['epsilon']))
        dump_json(OUT/'fgm_validation_selection.json',{'selected':selected,'candidates':candidates,'criterion':'clean validation Macro-F1; tie smaller epsilon; no test selection'})
        text+=f"\nFGM được chọn bằng validation: **{selected['condition']}**, epsilon **{selected['epsilon']}**, validation Macro-F1 **{selected['validation_macro_f1']:.4f}**.\n"
    text+='\n## Chênh lệch so với baseline và bootstrap\n\nCI bootstrap resample theo nhóm văn bản gốc trùng sau NFC/khoảng trắng; các ID cùng văn bản và 3 seed biến thể luôn đi chung nhóm. CI này không phản ánh độ biến thiên qua nhiều training seeds.\n\n| Model | Profile | Delta Macro-F1 | 95% CI |\n|---|---|---:|---|\n'
    for r in bootstrap:text+=f"| {r['model']} | {r['category']} | {r['delta_macro_f1']:+.4f} | [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] |\n"
    cohort_path=OUT/'ORIGINAL_TEST_COHORTS.md'
    if cohort_path.exists():text+='\n\n'+cohort_path.read_text()
    token_path=OUT/'TOKENIZATION_EFFECTS.md'
    if token_path.exists():text+='\n\n'+token_path.read_text()
    shared_path=OUT/'SHARED_ELIGIBLE_COMPARISON.md'
    if shared_path.exists():text+='\n\n'+shared_path.read_text()
    text+='\n## Cách diễn giải\n\nĐọc thêm coverage và changed-only F1 trong all_results.csv; mẫu không có mapping phù hợp được giữ nguyên, không được tính là thành công sửa noise. Controlled stress 10/20/30/50 là mức yêu cầu; tỷ lệ thực tế có thể khác do số vị trí đủ điều kiện, làm tròn và thay thế multi-unit.\n\nKết quả trên ViHSD gốc và natural cohorts dùng nhãn thật sẵn có; các cohort dùng marker NSW heuristic và có thể khác phân bố nhãn/chủ đề. Không suy diễn quan hệ nhân quả từ chênh lệch cohort. Giữ nguyên split chính thức có duplicate và báo riêng giới hạn này.\n\nKhông coi FGM chắc chắn tốt hơn hoặc augmentation chắc chắn hiệu quả. Kết luận cần bám bảng kết quả; cần nhiều training seeds và kiểm chứng ngôn ngữ trước khi tuyên bố tổng quát cho UGC hiện tại.\n'
    recovered=json.loads((Path(roots['augmentation-v2'])/'ROBUSTNESS_COMPLETE.json').read_text())
    if recovered.get('status')=='complete_recovered':
        text+=f"\n## Provenance của artifact augmentation\n\nTraining chạy đủ 5 epoch ở GPU notebook version 1, lưu đủ 33 model/profile evaluations, sau đó ERROR khi mở một optional pilot reference không được Kaggle gắn vào input. Giữ nguyên trạng thái ERROR gốc. Output đã được phục hồi và tính lại độc lập; phiên CPU kế tiếp chỉ xác minh/đóng gói, không train lại. Best weights SHA-256 `{recovered['best_model_sha256']}`. Source/log gốc và metadata phục hồi nằm cùng artifact. Code repo sau training đã thêm kiểm tra references trước mọi chuẩn bị dữ liệu/training.\n"
    for name in ['01_primary_comparison.png','02_stress_and_actual_rates.png']:text+=f'\n![{name}]({name})\n'
    (OUT/'ROBUSTNESS_REPORT.md').write_text(text)
    combined=(ROOT/'reports/noise_v2/NOISE_RESEARCH_V2.md').read_text()+'\n\n'+text
    renderer=MarkdownIt('commonmark').enable('table');body=renderer.render(combined)
    for folder in [ROOT/'reports/noise_v2',OUT]:
        for picture in folder.glob('*.png'):
            body=body.replace('src="'+picture.name+'"','src="data:image/png;base64,'+base64.b64encode(picture.read_bytes()).decode()+'"')
    page='<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ViHSD Robustness</title><style>body{font:16px/1.7 system-ui;color:#243546;background:#eef2f6;margin:0}main{max-width:1150px;margin:32px auto;padding:40px;background:white;border-top:6px solid #3979a8}h1{font-size:28px}h2{margin-top:32px}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce3ea;text-align:left}th{background:#eff4f8}code{background:#eff4f8;padding:2px 4px}@media(max-width:700px){main{padding:18px;margin:0}table{font-size:11px}}</style><main>'+body+'</main></html>'
    (ROOT/'reports/ROBUSTNESS.html').write_text(page)
    print(means.to_string(index=False))
    return means

if __name__=='__main__':
    roots={c:ROOT/f'artifacts/robustness/{c}/robustness_project' for c in ['augmentation-v2','fgm-small']}
    assert all((p/'ROBUSTNESS_COMPLETE.json').exists() for p in roots.values()),'Both primary v2 jobs required'
    run(roots)
